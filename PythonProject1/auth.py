from passlib.context import CryptContext

# Set up the secure Bcrypt hashing configuration layer
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Converts a raw password string into an unbreakable cryptographic hash."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Safely crosschecks an incoming login attempt against the stored database hash."""
    return pwd_context.verify(plain_password, hashed_password)
