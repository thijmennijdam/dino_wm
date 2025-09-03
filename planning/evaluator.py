import os
import torch
import imageio
import cv2
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device

        self.plot_full = False  # plot all frames or frames after frameskip

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        with torch.no_grad():
            i_z_obses, _ = self.wm.rollout(
                obs_0=trans_obs_0,
                act=actions,
            )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )

        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            
            # ADDED
            # Compute per-frame latent MSE between imagined visual latents and goal latents
            try:
                with torch.no_grad():
                    # encode goal obs latents
                    g_z = self.wm.encode_obs(trans_obs_g)["visual"]  # (b, 1, p, d)
                    i_z_vis = i_z_obses["visual"]  # (b, t, p, d)
                    # Broadcast goal across time and compute MSE per frame
                    # result: (b, t)
                    mse_per_frame = ((i_z_vis - g_z).pow(2).mean(dim=(2, 3)))
                    # Upsample to env frame-rate by repeating frames per frameskip
                    # shape to (b, t, 1) -> repeat -> (b, t*frameskip)
                    mse_rep = mse_per_frame.unsqueeze(2).repeat(
                        1, 1, self.frameskip
                    )
                    mse_rep = rearrange(mse_rep, "b t n -> b (t n)")
            except Exception:
                mse_rep = None
                mse_per_frame = None
            ###
                
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
                mse_rep=mse_rep,
                mse_steps=mse_per_frame,
            )

        return logs, successes, e_obses, e_states

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })

        return logs, successes

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename="", mse_rep=None, mse_steps=None
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    
                    # ADDED
                    # Overlay per-frame latent MSE if provided
                    if mse_rep is not None:
                        try:
                            if i < mse_rep.shape[1]:
                                mse_val = mse_rep[idx, i].item()

                                # Ensure HWC, uint8, contiguous
                                vis = frame
                                if hasattr(vis, "detach"):
                                    vis = vis.detach().cpu().numpy()
                                if vis.ndim == 2:  # grayscale -> 3-channel
                                    vis = np.stack([vis, vis, vis], axis=-1)
                                elif vis.ndim == 3 and vis.shape[-1] not in (1, 3):
                                    # If CHW slipped through, convert to HWC
                                    if vis.shape[0] in (1, 3):
                                        vis = np.transpose(vis, (1, 2, 0))

                                vis = (((np.clip(vis, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                                vis = np.ascontiguousarray(vis)  # <- key for OpenCV

                                cv2.putText(
                                    vis,
                                    f"MSE: {mse_val:.4f}",
                                    (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (49, 222, 49),
                                    2,
                                    cv2.LINE_AA,
                                )

                                # Back to [-1,1] float for the rest of your code
                                frame = (vis.astype(np.float32) / 255.0) * 2 - 1
                        except Exception:
                            pass
                        ###
                        
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        # Overlay latent MSE on imagined frames for PNGs (per time column)
        if mse_steps is not None:
            try:
                T = i_visuals.shape[1]  # number of time steps (columns)
                mse_steps = mse_steps[:, :T]  # align shapes

                i_list = []
                for b_idx in range(i_visuals.shape[0]):
                    frames_b = []
                    for t_idx in range(T):
                        img = i_visuals[b_idx, t_idx].detach().cpu()  # (C,H,W) tensor in [-1,1]
                        np_img = img.numpy()

                        # Convert (C,H,W) [-1,1] -> (H,W,C) uint8 [0,255]
                        vis = np.transpose(np_img, (1, 2, 0))  # HWC
                        vis = (((np.clip(vis, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                        vis = np.ascontiguousarray(vis)

                        # Get MSE value
                        mval = float(mse_steps[b_idx, t_idx].item())

                        # Draw clean black text (top-left corner)
                        cv2.putText(
                            vis,
                            f"MSE: {mval:.4f}",
                            (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (49, 222, 49),  # dark green
                            2,
                            cv2.LINE_AA,
                        )

                        # Convert back to [-1,1] float32 in CHW
                        vis_f = (vis.astype(np.float32) / 255.0) * 2 - 1
                        vis_f = np.transpose(vis_f, (2, 0, 1))  # CHW
                        frames_b.append(torch.from_numpy(vis_f))

                    i_list.append(torch.stack(frames_b, dim=0))  # (T,C,H,W)

                i_visuals = torch.stack(i_list, dim=0)  # (B,T,C,H,W)

            except Exception as e:
                print(f"Warning: failed to overlay MSE text: {e}")
                pass


        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )
