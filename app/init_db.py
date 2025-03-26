"""
Database initialization script.
"""
from app.db.database import init_db, get_db_session
from app.db.models import User
import uuid

def create_admin_user():
    """Create the admin user if it doesn't exist."""
    db = get_db_session()
    try:
        # Check if admin user exists
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            # Create admin user
            admin_user = User(
                id=uuid.uuid4(),
                username="admin",
                email="admin@example.com",
                password_hash="admin",  # This should be properly hashed in production
                is_active=True,
                is_admin=True
            )
            db.add(admin_user)
            db.commit()
            print("Admin user created successfully")
        else:
            print("Admin user already exists")
    finally:
        db.close()

def main():
    """Initialize the database and create admin user."""
    print("Initializing database...")
    init_db()
    print("Database initialized successfully")
    
    print("Creating admin user...")
    create_admin_user()

if __name__ == "__main__":
    main() 