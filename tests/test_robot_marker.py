# SPDX-License-Identifier: GPL-3.0-or-later
"""Run: MUJOCO_GL=egl .venv/bin/python -m unittest discover -s tests -v

Requires OpenCV in the test environment; the simulator itself does not need it.
"""
import unittest
from unittest.mock import patch

import cv2
import mujoco
import numpy as np

from climbing_sim import ClimbingSim, build_spec
from climbing_sim.robot_marker import (
    GEOM_NAME, MARKER_CENTRE, MARKER_SIZE, R_TRUNK_MARKER,
    VISUAL_HALF_THICKNESS, VISUAL_OFFSET,
)
from climbing_sim.wall import world_to_wall


class RobotMarkerTest(unittest.TestCase):
    def setUp(self):
        with patch("climbing_sim.scene.add_robot_marker"):
            self.baseline = ClimbingSim("N2")
        self.marked = ClimbingSim("N2")

    def tearDown(self):
        self.baseline.close()
        self.marked.close()

    def test_visual_transform_and_size(self):
        model = self.marked.model
        geom = model.geom(GEOM_NAME)
        self.assertEqual(int(geom.bodyid[0]), model.body("trunk").id)
        self.assertEqual(int(geom.contype[0]), 0)
        self.assertEqual(int(geom.conaffinity[0]), 0)
        np.testing.assert_array_equal(geom.size[:2] * 2, [0.042, 0.042])
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, geom.quat)
        np.testing.assert_allclose(rotation.reshape(3, 3), R_TRUNK_MARKER, atol=1e-15)
        front = geom.pos + R_TRUNK_MARKER[:, 2] * VISUAL_HALF_THICKNESS
        np.testing.assert_allclose(front, MARKER_CENTRE +
                                   R_TRUNK_MARKER[:, 2] * VISUAL_OFFSET, atol=1e-15)

    def test_original_model_and_dynamics_unchanged(self):
        old, new = self.baseline.model, self.marked.model
        self.assertEqual(new.ngeom, old.ngeom + 1)
        for count in ("nbody", "njnt", "nq", "nv", "nu", "nmesh", "ncam", "nsensor", "neq"):
            self.assertEqual(getattr(old, count), getattr(new, count), count)
        for field in ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos",
                      "body_iquat", "jnt_pos", "jnt_axis", "jnt_range", "dof_damping",
                      "dof_armature", "dof_frictionloss", "actuator_gear", "actuator_gainprm",
                      "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange",
                      "qpos0", "mesh_vert", "mesh_face", "mesh_scale", "cam_pos", "cam_quat",
                      "cam_intrinsic", "sensor_type", "sensor_objid"):
            np.testing.assert_array_equal(getattr(old, field), getattr(new, field), err_msg=field)
        marker_id = new.geom(GEOM_NAME).id
        for field in ("geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size",
                      "geom_contype", "geom_conaffinity", "geom_friction", "geom_solref",
                      "geom_solimp", "geom_dataid", "geom_group", "geom_matid", "geom_rgba"):
            np.testing.assert_array_equal(getattr(old, field),
                                          np.delete(getattr(new, field), marker_id, axis=0),
                                          err_msg=field)
        a, b = self.baseline.reset_hanging(), self.marked.reset_hanging()
        np.testing.assert_array_equal(a["q"], b["q"])
        target = a["q"].copy()
        for step in range(100):
            command = target.copy()
            command[-1] += 0.05 * np.sin(step * 0.1)
            self.baseline.step(command)
            self.marked.step(command)
            np.testing.assert_array_equal(self.baseline.data.qpos, self.marked.data.qpos)
            np.testing.assert_array_equal(self.baseline.data.qvel, self.marked.data.qvel)
            np.testing.assert_array_equal(self.baseline.data.actuator_force,
                                          self.marked.data.actuator_force)

    def test_id5_detected_with_correct_corner_orientation_in_both_cameras(self):
        sim = self.marked
        sim.reset_hanging()
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        # Canonical OpenCV corner order: top-left, top-right, bottom-right, bottom-left.
        half = MARKER_SIZE / 2
        local = np.array([[-half, half, 0], [half, half, 0],
                          [half, -half, 0], [-half, -half, 0]])
        trunk = sim.data.body("trunk")
        points_trunk = local @ R_TRUNK_MARKER.T + MARKER_CENTRE
        points_world = points_trunk @ trunk.xmat.reshape(3, 3).T + trunk.xpos
        for name, camera in sim.cameras.items():
            with self.subTest(camera=name):
                image = sim.render(name)
                corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
                self.assertIsNotNone(ids)
                self.assertIn(5, ids.ravel())
                actual = corners[list(ids.ravel()).index(5)][0]
                expected = camera.project_wall(world_to_wall(points_world))
                # Rasterisation of the small marker in the unmodified wide camera views.
                np.testing.assert_allclose(actual, expected, atol=1.5, rtol=0)

    def test_no_robot_means_no_robot_marker(self):
        spec = build_spec(robot=False)
        self.assertNotIn(GEOM_NAME, [g.name for g in spec.geoms])


if __name__ == "__main__":
    unittest.main()
