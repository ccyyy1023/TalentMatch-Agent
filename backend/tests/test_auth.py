from app.schemas import UserCreate
from app.services.auth import AuthService


def test_password_hash_is_salted_and_verifiable(tmp_path):
    first = AuthService.hash_password("strong-password")
    second = AuthService.hash_password("strong-password")
    assert first != second
    assert AuthService.verify_password("strong-password", first)
    assert not AuthService.verify_password("wrong-password", first)


def test_session_is_persistent_and_revocable(tmp_path):
    database = tmp_path / "auth.db"
    service = AuthService(database)
    user = service.create_user(UserCreate(username="Recruiter.One", password="strong-password", role="recruiter"))
    login = service.login("recruiter.one", "strong-password")
    assert login is not None
    assert login.access_token not in database.read_bytes().decode("utf-8", errors="ignore")
    assert AuthService(database).authenticate(login.access_token).id == user.id
    service.logout(login.access_token)
    assert service.authenticate(login.access_token) is None


def test_audit_record_keeps_actor_and_business_action(tmp_path):
    service = AuthService(tmp_path / "audit.db")
    user = service.create_user(UserCreate(username="admin.user", password="strong-password", role="admin"))
    service.record_audit(user, "decide", "candidate", "candidate-1", {"decision": "advance"})
    record = service.list_audit()[0]
    assert record.actor_username == "admin.user"
    assert record.resource_id == "candidate-1"
    assert record.detail["decision"] == "advance"
