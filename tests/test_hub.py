"""Day 13 tests: Hugging Face publishing helpers — offline.

A FakeHfApi records what WOULD be created / uploaded, so the tests check the important promises
without a network or a token:
  - the public bundle never contains a demo user's ratings (MovieLens licence),
  - the published bundle still loads and serves,
  - the Space gets the code + its Dockerfile + a README with the Space header, and BUNDLE_REPO.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recsys import hub  # noqa: E402
from recsys.serving import Engine  # noqa: E402
from recsys.synthetic import FakeEncoder, build_synthetic_bundle  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hub")
    build_synthetic_bundle(tmp / "bundle", work_dir=tmp)
    return tmp / "bundle"


class FakeHfApi:
    """Stands in for huggingface_hub.HfApi; copies every uploaded folder so tests can inspect it."""
    calls: list = []

    def __init__(self, token=None):
        pass

    def create_repo(self, repo_id, **kw):
        FakeHfApi.calls.append(("create_repo", repo_id, kw))
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(self, *, repo_id, folder_path, **kw):
        snap = Path(FakeHfApi.snapshot_dir) / repo_id.replace("/", "__")
        if snap.exists():
            shutil.rmtree(snap)
        shutil.copytree(folder_path, snap)
        FakeHfApi.calls.append(("upload_folder", repo_id, {**kw, "snapshot": snap}))

    def add_space_variable(self, repo_id, key, value):
        FakeHfApi.calls.append(("add_space_variable", repo_id, {key: value}))


@pytest.fixture
def fake_hf(monkeypatch, tmp_path):
    import huggingface_hub
    FakeHfApi.calls = []
    FakeHfApi.snapshot_dir = tmp_path / "uploaded"
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    return FakeHfApi


def test_stage_public_bundle_drops_demo_users(bundle, tmp_path):
    assert (bundle / "demo_history.parquet").exists()            # the local bundle has them
    man = hub.stage_public_bundle(bundle, tmp_path / "pub")
    files = {p.name for p in (tmp_path / "pub").iterdir()}
    assert files == set(hub.BUNDLE_FILES)                         # no demo_history / demo_truth
    assert man["n_demo_users"] == 0 and man["public"] is True
    assert json.loads((tmp_path / "pub" / "bundle.json").read_text())["n_demo_users"] == 0
    eng = Engine.load(tmp_path / "pub", encoder=FakeEncoder())    # still a working bundle
    assert len(eng.demo_users) == 0 and eng.cold_start([10, 20, 30], k=3)["model"] == "warm ranker"


def test_stage_public_bundle_keeps_demo_users_only_when_asked(bundle, tmp_path):
    man = hub.stage_public_bundle(bundle, tmp_path / "all", include_demo_users=True)
    assert (tmp_path / "all" / "demo_history.parquet").exists() and man["public"] is False


def test_stage_refuses_an_incomplete_bundle(bundle, tmp_path):
    shutil.copytree(bundle, tmp_path / "broken")
    (tmp_path / "broken" / "arrays.npz").unlink()
    with pytest.raises(FileNotFoundError, match="arrays.npz"):
        hub.stage_public_bundle(tmp_path / "broken", tmp_path / "out")


def test_upload_bundle_publishes_card_and_no_ratings(bundle, fake_hf):
    url = hub.upload_bundle(bundle, "alice/semantic-recsys-bundle", code_url="https://github.com/x/y")
    assert url == "https://huggingface.co/alice/semantic-recsys-bundle"
    kinds = [c[0] for c in fake_hf.calls]
    assert kinds == ["create_repo", "upload_folder"]
    assert fake_hf.calls[0][2]["repo_type"] == "model" and fake_hf.calls[0][2]["private"] is False
    snap = fake_hf.calls[1][2]["snapshot"]
    names = {p.name for p in snap.iterdir()}
    assert "README.md" in names and not names & set(hub.DEMO_FILES)
    card = (snap / "README.md").read_text(encoding="utf-8")
    assert "No individual ratings are included" in card and "https://github.com/x/y" in card


def test_stage_space_has_code_dockerfile_and_space_header(tmp_path):
    hub.stage_space(ROOT, tmp_path / "space", title="Demo", code_url="https://github.com/x/y",
                    commit="abcdef1234")
    s = tmp_path / "space"
    for f in ("Dockerfile", "README.md", "pyproject.toml", "requirements-serve.txt",
              "requirements-ui.txt", "deploy/start_space.sh", "app/streamlit_app.py",
              "src/recsys/api.py", "src/recsys/serving.py", "src/recsys/hub.py"):
        assert (s / f).exists(), f
    assert not list(s.rglob("__pycache__"))
    readme = (s / "README.md").read_text(encoding="utf-8")
    header = readme.split("---")[1]
    assert "sdk: docker" in header and "app_port: 7860" in header and "title: Demo" in header
    assert "abcdef1" in readme and "{{" not in readme
    assert (s / "Dockerfile").read_text() == (ROOT / "deploy/hf_space/Dockerfile").read_text()


def test_deploy_space_sets_bundle_repo_and_uploads(fake_hf):
    url = hub.deploy_space(ROOT, "alice/semantic-recsys-demo", "alice/semantic-recsys-bundle",
                           code_url="https://github.com/x/y")
    assert url == "https://huggingface.co/spaces/alice/semantic-recsys-demo"
    create = [c for c in fake_hf.calls if c[0] == "create_repo"][0]
    assert create[2]["repo_type"] == "space" and create[2]["space_sdk"] == "docker"
    vars_ = {k: v for c in fake_hf.calls if c[0] == "add_space_variable" for k, v in c[2].items()}
    assert vars_["BUNDLE_REPO"] == "alice/semantic-recsys-bundle"
    up = [c for c in fake_hf.calls if c[0] == "upload_folder"][0]
    assert up[2]["repo_type"] == "space" and (up[2]["snapshot"] / "Dockerfile").exists()


def test_space_dockerfile_and_start_script_agree():
    """The Dockerfile copies what start_space.sh runs, as user 1000, exposing the README's port."""
    df = (ROOT / "deploy/hf_space/Dockerfile").read_text()
    sh = (ROOT / "deploy/start_space.sh").read_text()
    assert "useradd -m -u 1000 user" in df and "USER user" in df and "EXPOSE 7860" in df
    assert "deploy/start_space.sh" in df and "app ./app" in df and "src ./src" in df
    assert "python -m recsys.hub download" in sh and "${PORT:-7860}" in sh
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n", str(ROOT / "deploy/start_space.sh")]).returncode == 0


def test_stage_space_converts_windows_line_endings(tmp_path):
    root = tmp_path / "repo"
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(".git", "data", "reports", "tests",
                                                               "notebooks", "__pycache__"))
    sh = root / "deploy" / "start_space.sh"
    sh.write_bytes(sh.read_bytes().replace(b"\n", b"\r\n"))          # as a CRLF checkout would
    hub.stage_space(root, tmp_path / "space", title="T", code_url="u", commit="c")
    assert b"\r\n" not in (tmp_path / "space" / "deploy" / "start_space.sh").read_bytes()
    assert b"\r\n" not in (tmp_path / "space" / "Dockerfile").read_bytes()
