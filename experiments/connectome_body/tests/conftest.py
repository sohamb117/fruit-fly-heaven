import pytest

from connectome_body.util import atomic_json, digest_file


@pytest.fixture
def compatibility_manifest_factory(tmp_path):
    """Native MuJoCo fixture; explicitly not evidence about worm/fish biology."""

    def create(body="worm"):
        root = tmp_path / body
        root.mkdir(exist_ok=True)
        xml = root / "model.xml"
        xml.write_text("""<mujoco model="compatibility-test-fixture">
  <option gravity="0 0 0" timestep="0.002"/>
  <default><joint damping="0.2"/><geom density="1000" contype="0" conaffinity="0"/></default>
  <worldbody><body name="root" pos="0 0 0.1"><freejoint/>
    <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.006"/>
    <body name="tail" pos="0.05 0 0"><joint name="bend" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.005"/>
    </body></body></worldbody>
  <actuator><motor name="bend_motor" joint="bend" gear="0.0001"
    ctrllimited="true" ctrlrange="-1 1"/></actuator>
</mujoco>""")
        manifest = root / "manifest.json"
        atomic_json(
            manifest,
            {
                "schema": "mujoco-embodiment-v1",
                "body": body,
                "species": "software-fixture",
                "source": {"is_synthetic": True},
                "files": {"model.xml": digest_file(xml)},
                "model": "model.xml",
                "root_body": "root",
                "body_length": 0.1,
                "forward_axis": [1, 0, 0],
                "control_dt": 0.01,
                "evidence": "software_fixture_only",
            },
        )
        return manifest

    return create
