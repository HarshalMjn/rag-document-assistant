from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.config.settings import settings


async def test_migrations_created_tables():
    # The container runs `alembic upgrade head` on startup, so the schema should already exist.
    engine = create_async_engine(settings.DATABASE_URI, poolclass=NullPool)
    async with engine.connect() as conn:
        tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
    await engine.dispose()

    assert {"documents", "document_versions", "conversations", "messages"} <= set(tables)
