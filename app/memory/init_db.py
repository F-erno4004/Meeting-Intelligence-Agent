from sqlalchemy import text

from app.memory.database import Base, engine

# Import models so SQLAlchemy registers all tables.
from app.memory import models


def initialize_database():
    print("Initializing Neon database...")

    try:
        # Enable pgvector before creating vector columns and indexes.
        with engine.begin() as connection:
            connection.execute(
                text("CREATE EXTENSION IF NOT EXISTS vector")
            )

        # Create all registered tables and indexes.
        Base.metadata.create_all(bind=engine)

        print("Database schema created successfully!")

        print("\nCreated tables:")
        for table_name in Base.metadata.tables:
            print(f"- {table_name}")

    except Exception as error:
        print(f"Database initialization failed: {error}")
        raise

    finally:
        engine.dispose()


if __name__ == "__main__":
    initialize_database()
