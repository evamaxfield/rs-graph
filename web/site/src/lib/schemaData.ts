// Schema reference data, derived directly from rs_graph/db/models.py.
// Every table also carries created_datetime / updated_datetime timestamps,
// omitted below to avoid repeating them 27 times.

export interface TableField {
  name: string;
  type: string;
}

export interface TableDef {
  name: string;
  docstring: string;
  fields: TableField[];
}

export interface FamilyDef {
  slug: string;
  title: string;
  reason: string;
  tables: TableDef[];
}

export const families: FamilyDef[] = [
  {
    slug: 'documents',
    title: 'Documents',
    reason: "Everything hanging directly off a paper's own identity.",
    tables: [
      {
        name: 'Document',
        docstring: 'Stores paper, report, or other academic document details.',
        fields: [
          { name: 'doi', type: 'str' },
          { name: 'open_alex_id', type: 'str' },
          { name: 'title', type: 'str' },
          { name: 'publication_date', type: 'date' },
          { name: 'cited_by_count', type: 'int' },
          {
            name: 'fwci',
            type: 'float | None -- Field-Weighted Citation Impact: citation count normalized against the global average for same-year, same-field documents (1.0 = average)',
          },
          { name: 'citation_normalized_percentile', type: 'float | None' },
          { name: 'document_type', type: 'str' },
          { name: 'is_open_access', type: 'bool' },
          { name: 'open_access_status', type: 'str' },
          { name: 'primary_location_id', type: 'int | None (FK: location)' },
          { name: 'best_open_access_location_id', type: 'int | None (FK: location)' },
        ],
      },
      {
        name: 'DocumentAbstract',
        docstring: 'Stores the abstract for a document.',
        fields: [
          { name: 'document_id', type: 'int (FK: document)' },
          { name: 'content', type: 'str | None' },
        ],
      },
      {
        name: 'DocumentAlternateDOI',
        docstring:
          'Stores alternate DOIs for a document -- DOIs previously associated with a document that resolved to a more recent version.',
        fields: [
          { name: 'document_id', type: 'int (FK: document)' },
          { name: 'doi', type: 'str' },
        ],
      },
      {
        name: 'DocumentTopic',
        docstring: 'Stores the connection between a document and a topic.',
        fields: [
          { name: 'document_id', type: 'int (FK: document)' },
          { name: 'topic_id', type: 'int (FK: topic)' },
          { name: 'score', type: 'float' },
        ],
      },
      {
        name: 'DocumentSoftwareMention',
        docstring:
          'Stores software mentions found in an academic paper (from the SoftCite dataset).',
        fields: [
          { name: 'document_id', type: 'int (FK: document)' },
          { name: 'softcite_mention_id', type: 'str' },
          { name: 'software_name', type: 'str' },
          { name: 'software_name_normalized', type: 'str' },
          { name: 'mention_context', type: 'str | None' },
        ],
      },
      {
        name: 'Topic',
        docstring:
          "Stores the basic information for a topic. Topics come from OpenAlex's 4-level hierarchy, broadest to narrowest: domain (4 values) > field (26 values, e.g. Genetics) > subfield > topic itself (most granular, this row's own name).",
        fields: [
          { name: 'open_alex_id', type: 'str' },
          { name: 'name', type: 'str' },
          { name: 'field_name', type: 'str' },
          { name: 'field_open_alex_id', type: 'str' },
          { name: 'subfield_name', type: 'str' },
          { name: 'subfield_open_alex_id', type: 'str' },
          { name: 'domain_name', type: 'str' },
          { name: 'domain_open_alex_id', type: 'str' },
        ],
      },
    ],
  },
  {
    slug: 'people-and-institutions',
    title: 'People & Institutions',
    reason: 'Author-side entities and their document-scoped roles.',
    tables: [
      {
        name: 'Researcher',
        docstring: 'Stores researcher details.',
        fields: [
          { name: 'open_alex_id', type: 'str' },
          { name: 'orcid', type: 'str | None' },
          { name: 'name', type: 'str' },
          { name: 'works_count', type: 'int' },
          { name: 'cited_by_count', type: 'int' },
          { name: 'h_index', type: 'int' },
          { name: 'i10_index', type: 'int' },
          { name: 'two_year_mean_citedness', type: 'float' },
        ],
      },
      {
        name: 'Institution',
        docstring: 'Stores institution details.',
        fields: [
          { name: 'open_alex_id', type: 'str' },
          { name: 'name', type: 'str' },
          { name: 'country_code', type: 'str | None' },
          { name: 'institution_type', type: 'str | None' },
          { name: 'ror', type: 'str | None' },
        ],
      },
      {
        name: 'DocumentContributor',
        docstring: 'Stores the connection between a researcher and a document.',
        fields: [
          { name: 'researcher_id', type: 'int (FK: researcher)' },
          { name: 'document_id', type: 'int (FK: document)' },
          { name: 'position', type: 'str' },
          { name: 'is_corresponding', type: 'bool' },
        ],
      },
      {
        name: 'DocumentContributorInstitution',
        docstring: 'Stores the connection between a researcher, document, and institution.',
        fields: [
          { name: 'document_contributor_id', type: 'int (FK: document_contributor)' },
          { name: 'institution_id', type: 'int (FK: institution)' },
        ],
      },
    ],
  },
  {
    slug: 'funding',
    title: 'Funding',
    reason: 'Small, self-contained 3-table family.',
    tables: [
      {
        name: 'Funder',
        docstring: 'Stores the basic information for a funding source (e.g. NSF, NIH).',
        fields: [
          { name: 'open_alex_id', type: 'str' },
          { name: 'name', type: 'str' },
        ],
      },
      {
        name: 'FundingInstance',
        docstring: 'Stores the basic information for a single funding instance (e.g. grant).',
        fields: [
          { name: 'funder_id', type: 'int (FK: funder)' },
          { name: 'award_id', type: 'str' },
        ],
      },
      {
        name: 'DocumentFundingInstance',
        docstring: 'Stores the connection between a document and a funding instance.',
        fields: [
          { name: 'document_id', type: 'int (FK: document)' },
          { name: 'funding_instance_id', type: 'int (FK: funding_instance)' },
        ],
      },
    ],
  },
  {
    slug: 'repositories-and-code',
    title: 'Repositories & Code',
    reason:
      'The repo-side mirror of the Documents page. No raw source file contents are stored here: ' +
      'RepositoryFile is path/size/language metadata only, and RepositoryImport / RepositoryDependency ' +
      'are extracted library-name signal, not code. The only free-text field is RepositoryReadme.content.',
    tables: [
      {
        name: 'Repository',
        docstring: 'Stores the basic information for a repository.',
        fields: [
          { name: 'code_host_id', type: 'int (FK: code_host)' },
          { name: 'owner', type: 'str' },
          { name: 'name', type: 'str' },
          { name: 'description', type: 'str | None' },
          { name: 'is_fork', type: 'bool' },
          { name: 'forks_count', type: 'int' },
          { name: 'stargazers_count', type: 'int' },
          { name: 'watchers_count', type: 'int' },
          { name: 'open_issues_count', type: 'int' },
          { name: 'commits_count', type: 'int | None' },
          { name: 'size_kb', type: 'int' },
          { name: 'topics', type: 'str | None' },
          { name: 'primary_language', type: 'str | None' },
          { name: 'default_branch', type: 'str | None' },
          { name: 'license', type: 'str | None' },
          { name: 'creation_datetime', type: 'datetime' },
          { name: 'last_pushed_datetime', type: 'datetime' },
        ],
      },
      {
        name: 'CodeHost',
        docstring: 'Stores the basic information for a code host (e.g. GitHub, GitLab).',
        fields: [{ name: 'name', type: 'str' }],
      },
      {
        name: 'RepositoryReadme',
        docstring: 'Stores the readme for a repository.',
        fields: [
          { name: 'repository_id', type: 'int (FK: repository)' },
          { name: 'content', type: 'str | None' },
        ],
      },
      {
        name: 'RepositoryLanguage',
        docstring: 'Stores the connection between a repository and a language.',
        fields: [
          { name: 'repository_id', type: 'int (FK: repository)' },
          { name: 'language', type: 'str' },
          { name: 'bytes_of_code', type: 'int' },
        ],
      },
      {
        name: 'RepositoryFile',
        docstring: 'Stores the connection between a repository and a file.',
        fields: [
          { name: 'repository_id', type: 'int (FK: repository)' },
          { name: 'path', type: 'str' },
          { name: 'tree_type', type: 'str' },
          { name: 'bytes_of_code', type: 'int' },
        ],
      },
      {
        name: 'RepositoryImport',
        docstring:
          'Stores software libraries imported in repository source code (extracted via eil).',
        fields: [
          { name: 'repository_id', type: 'int (FK: repository)' },
          { name: 'software_name', type: 'str' },
          { name: 'software_name_normalized', type: 'str' },
          { name: 'file_paths', type: 'str | None -- semicolon-separated paths' },
        ],
      },
      {
        name: 'RepositoryDependency',
        docstring:
          'Stores dependencies declared in repository manifests (extracted via git-pkgs).',
        fields: [
          { name: 'repository_id', type: 'int (FK: repository)' },
          { name: 'software_name', type: 'str' },
          { name: 'software_name_normalized', type: 'str' },
          { name: 'version_spec', type: 'str | None' },
          { name: 'ecosystem', type: 'str | None' },
          { name: 'manifest_paths', type: 'str | None -- semicolon-separated paths' },
          { name: 'dependency_type', type: 'str | None' },
        ],
      },
      {
        name: 'RepositoryContributor',
        docstring: 'Stores the connection between a repository and a contributor.',
        fields: [
          { name: 'repository_id', type: 'int (FK: repository)' },
          { name: 'developer_account_id', type: 'int (FK: developer_account)' },
        ],
      },
    ],
  },
  {
    slug: 'developer-accounts-and-matching',
    title: 'Developer Accounts & Author-Developer Matching',
    reason:
      'The ML-matching boundary: predictive_model_* fields make this qualitatively different from the rest of People.',
    tables: [
      {
        name: 'DeveloperAccount',
        docstring:
          'Stores the basic information for a developer account (e.g. GitHub, GitLab).',
        fields: [
          { name: 'code_host_id', type: 'int (FK: code_host)' },
          { name: 'username', type: 'str' },
          { name: 'name', type: 'str | None' },
          { name: 'email', type: 'str | None' },
        ],
      },
      {
        name: 'ResearcherDeveloperAccountLink',
        docstring: 'Stores the connection between a researcher and a developer account.',
        fields: [
          { name: 'researcher_id', type: 'int (FK: researcher)' },
          { name: 'developer_account_id', type: 'int (FK: developer_account)' },
          { name: 'predictive_model_name', type: 'str | None' },
          { name: 'predictive_model_version', type: 'str | None' },
          {
            name: 'predictive_model_confidence',
            type: "float | None -- see Home's 'Using this dataset responsibly'",
          },
          { name: 'last_snowball_processed_datetime', type: 'datetime | None' },
        ],
      },
    ],
  },
  {
    slug: 'article-repository-links',
    title: 'Article-Repository Links',
    reason: 'The single relationship the whole project exists to produce.',
    tables: [
      {
        name: 'DocumentRepositoryLink',
        docstring: 'Stores the connection between a document and a repository.',
        fields: [
          { name: 'document_id', type: 'int (FK: document)' },
          { name: 'repository_id', type: 'int (FK: repository)' },
          { name: 'dataset_source_id', type: 'int (FK: dataset_source)' },
          { name: 'iteration', type: 'int | None' },
          { name: 'predictive_model_name', type: 'str | None' },
          { name: 'predictive_model_version', type: 'str | None' },
          {
            name: 'predictive_model_confidence',
            type: "float | None -- see Home's 'Using this dataset responsibly'",
          },
        ],
      },
      {
        name: 'DatasetSource',
        docstring: 'Stores the basic information for a dataset source.',
        fields: [{ name: 'name', type: 'str' }],
      },
    ],
  },
  {
    slug: 'provenance-and-internals',
    title: 'Provenance & Internals',
    reason: 'OpenAlex-sourced bibliographic plumbing -- real fields, low-traffic.',
    tables: [
      {
        name: 'Source',
        docstring: 'Stores the basic information about a possible document source.',
        fields: [
          { name: 'name', type: 'str' },
          { name: 'open_alex_id', type: 'str' },
          { name: 'source_type', type: 'str' },
          { name: 'host_organization_name', type: 'str | None' },
          { name: 'host_organization_open_alex_id', type: 'str | None' },
        ],
      },
      {
        name: 'Location',
        docstring: "Stores the basic information about a document's possible location.",
        fields: [
          { name: 'landing_page_url', type: 'str | None' },
          { name: 'pdf_url', type: 'str | None' },
          { name: 'source_id', type: 'int | None (FK: source)' },
          { name: 'is_open_access', type: 'bool' },
          { name: 'license', type: 'str | None' },
          { name: 'version', type: 'str | None' },
        ],
      },
    ],
  },
];
