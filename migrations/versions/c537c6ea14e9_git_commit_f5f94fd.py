"""Git Commit: f5f94fd

Revision ID: c537c6ea14e9
Revises: 
Create Date: 2024-03-29 16:45:14.053990

"""
from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c537c6ea14e9'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('dataset_source',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.create_table('funder',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('open_alex_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('open_alex_id')
    )
    op.create_table('institution',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('open_alex_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('ror', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('open_alex_id')
    )
    op.create_table('researcher',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('open_alex_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('works_count', sa.Integer(), nullable=False),
    sa.Column('cited_by_count', sa.Integer(), nullable=False),
    sa.Column('h_index', sa.Integer(), nullable=False),
    sa.Column('i10_index', sa.Integer(), nullable=False),
    sa.Column('two_year_mean_citedness', sa.Float(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('open_alex_id')
    )
    op.create_table('topic',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('open_alex_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('field_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('field_open_alex_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('subfield_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('subfield_open_alex_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('domain_name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('domain_open_alex_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('open_alex_id')
    )
    op.create_table('document',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('doi', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('open_alex_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('title', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('publication_date', sa.Date(), nullable=False),
    sa.Column('cited_by_count', sa.Integer(), nullable=False),
    sa.Column('cited_by_percentile_year_min', sa.Integer(), nullable=False),
    sa.Column('cited_by_percentile_year_max', sa.Integer(), nullable=False),
    sa.Column('abstract', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('dataset_source_id', sa.Integer(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['dataset_source_id'], ['dataset_source.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('doi')
    )
    op.create_table('funding_instance',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('funder_id', sa.Integer(), nullable=False),
    sa.Column('award_id', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['funder_id'], ['funder.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('funder_id', 'award_id')
    )
    op.create_table('document_funding_instance',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('funding_instance_id', sa.Integer(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['document_id'], ['document.id'], ),
    sa.ForeignKeyConstraint(['funding_instance_id'], ['funding_instance.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'funding_instance_id')
    )
    op.create_table('document_topic',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('topic_id', sa.Integer(), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['document_id'], ['document.id'], ),
    sa.ForeignKeyConstraint(['topic_id'], ['topic.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'topic_id')
    )
    op.create_table('researcher_document',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('researcher_id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('position', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
    sa.Column('is_corresponding', sa.Boolean(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['document_id'], ['document.id'], ),
    sa.ForeignKeyConstraint(['researcher_id'], ['researcher.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('researcher_id', 'document_id')
    )
    op.create_table('researcher_document_institution',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('researcher_document_id', sa.Integer(), nullable=False),
    sa.Column('institution_id', sa.Integer(), nullable=False),
    sa.Column('created_datetime', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.Column('updated_datetime', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['institution_id'], ['institution.id'], ),
    sa.ForeignKeyConstraint(['researcher_document_id'], ['researcher_document.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('researcher_document_id', 'institution_id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('researcher_document_institution')
    op.drop_table('researcher_document')
    op.drop_table('document_topic')
    op.drop_table('document_funding_instance')
    op.drop_table('funding_instance')
    op.drop_table('document')
    op.drop_table('topic')
    op.drop_table('researcher')
    op.drop_table('institution')
    op.drop_table('funder')
    op.drop_table('dataset_source')
    # ### end Alembic commands ###