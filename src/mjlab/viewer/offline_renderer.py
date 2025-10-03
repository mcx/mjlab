import mujoco
import mujoco_warp as mjwarp
import numpy as np

from mjlab.viewer.viewer_config import ViewerConfig


class OfflineRenderer:
  def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: ViewerConfig):
    self._cfg = cfg
    self._model = model
    self._data = data
    # self._data = mujoco.MjData(model)

    self._model.vis.global_.offheight = cfg.height
    self._model.vis.global_.offwidth = cfg.width
    self._model.stat.extent = 2.0

    if not cfg.enable_shadows:
      self._model.light_castshadow[:] = False
    if not cfg.enable_reflections:
      self._model.mat_reflectance[:] = 0.0

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(self._model, camera)
    # camera.distance = 50.0
    self._cam = camera

    self._renderer: mujoco.Renderer | None = None
    self._opt = mujoco.MjvOption()
    self._pert = mujoco.MjvPerturb()
    self._catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

  @property
  def renderer(self) -> mujoco.Renderer:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    return self._renderer

  def initialize(self) -> None:
    if self._renderer is not None:
      raise RuntimeError(
        "Renderer is already initialized. Call 'close()' first to reinitialize."
      )
    self._renderer = mujoco.Renderer(
      model=self._model, height=self._cfg.height, width=self._cfg.width
    )

  def update(self, data: mjwarp.Data) -> None:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    self._data.qpos[:] = data.qpos[0].cpu().numpy()
    self._data.qvel[:] = data.qvel[0].cpu().numpy()
    mujoco.mj_forward(self._model, self._data)

    # self._cam.azimuth = np.pi * np.cos(data.time[0].cpu().numpy() * 2.0 * np.pi)

    self._renderer.update_scene(self._data, camera=self._cam)

    # for i in range(min(data.nworld, 32)):
    #   self._data.qpos[:] = data.qpos[i].cpu().numpy()
    #   self._data.qvel[:] = data.qvel[i].cpu().numpy()
    #   mujoco.mj_forward(self._model, self._data)
    #   mujoco.mjv_addGeoms(
    #     self._model,
    #     self._data,
    #     self._opt,
    #     self._pert,
    #     self._catmask.value,
    #     self._renderer.scene,
    #   )

  def render(self) -> np.ndarray:
    if self._renderer is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    return self._renderer.render()

  def close(self) -> None:
    if self._renderer is not None:
      self._renderer.close()
      self._renderer = None
