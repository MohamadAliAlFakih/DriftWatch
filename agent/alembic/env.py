"""Alembic environment.

Reads ``agent_database_url`` from the agent's settings, strips the ``+asyncpg``
driver suffix (Alembic's online runner is sync — uses psycopg under the hood),
and runs migrations against the metadata declared in ``app.db.base.Base``.

Importing ``app.db.models`` is necessary even though the symbol isn't used —
the import side-effect is what registers the model classes against
``Base.metadata`` so autogenerate sees them.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.db import models  # noqa: F401  -- side-effect import; registers tables
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# build sync URL for Alembic — strip asyncpg driver suffix so psycopg drives migrations
settings = get_settings()
sync_url = settings.agent_database_url.replace("+asyncpg", "")
config.set_main_option("sqlalchemy.url", sync_url)

target_metadata = Base.metadata


# emit SQL to stdout for inspection without connecting to a DB
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# run migrations against a live DB
def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
