# Research Software Graph

🚀 [Rendered Interactive Graph](https://evamaxfield.github.io/rs-graph/) 🚀

Graphing the dependencies (and dependents) of research software and their contributors.

## Setup

* Install [Just](https://github.com/casey/just#packages)
* Create a new Conda / Mamba Env (Python 3.11 preferred)
* Install [pylbfgs](https://github.com/larsmans/pylbfgs): `conda install -c fgregg pylbfgs`
* Install lib: `just install` (or `pip install .`)

### Data Download

There is a lot of pre-processed data stored on a GCS bucket. To download it, run:

```bash
rs-graph-data download
```

This will always download the latest version of the data (as pushed by @evamaxfield).

## Order of Data Operations

If this project was starting from scratch, the order of operations to recreate our work
would be:

**Note:** at each step of the process, the data or model is cached
to the installed package directory (`rs_graph.data.files`).

1. Get all data sources using the command: `rs-graph-sources get-all`
2. Options
    1. Get all recursive upstream dependencies for the repos:
        `rs-graph-enrichment get-upstreams`
    2. Get all of the extended paper details for the papers:
        `rs-graph-enrichment get-extended-paper-details`
    3. Get top 30 contributors to each repository:
        `rs-graph-enrichment get-repo-contributors`
    4.  Train github users dedupe model:
        `rs-graph-modeling train-developer-deduper`
    5. Train author to github user linker model:
        `rs-graph-modeling train-author-developer-linker`


## GitHub User and Author Entity Matching Model

### IRR Annotation

From the 2023-12-12 upload of the full dataset, we randomly sampled 100 rows to
use for three-way annotator-agreement and alignment.

Annotation criteria for entity matches are:

1. `dev_details` name, username, or email is the same or incredibly
    similar to the `author_details` name. specifically:

    1.  if the provided name in the `dev_details` is the same name as the
        author name from the `author_details` (or incredibly similar)
    
    or:

    2.  if no name is provided in the `dev_details` but the GitHub username
        or the email within `dev_details` looks similar to the
        author name from `author_details`

2.  there is no co-author (names) or co-contributor (usernames) which look
    like a potentially better match.

3. if there is uncertainty, err on the side of `False`

#### Examples

##### No Match -- No Name Provided, GitHub Username or Email not Similar to Author Name

| dev_details 	| author_details 	| match 	|
|---	|---	|---	|
| username: ideas-man;<br>name: None;<br>email: None;<br>repos: https://github.com/DLR-RM/BlenderProc;<br>co_contributors: themasterlink, cornerfarmer, MartinSmeyer, mayman99, wboerdijk, joe3141, MarkusKnauer, Sebastian-Jung, 5trobl, thodan, probabilisticrobotics, mansoorcheema, DavidRisch, apenzko, abahnasy, maximilianmuehlbauer, moizsajid, Victorlouisdg, hansaskov, wangg12, MarwinNumbers, harinandan1995, neixlo, zzilch, cuteday, andrewyguo, jascase901, HectorAnadon, beekama; 	| name: Klaus H. Strobl;<br>repos: https://github.com/DLR-RM/BlenderProc;<br>co_authors: Martin Sundermeyer, Rudolph Alexander Triebel, Dominik Winkelbauer, Wout Boerdijk, Matthias Humt, Markus C. Knauer, Maximilian Denninger; 	| FALSE 	|

This row is labeled `False` because there is no provided name in `dev_details` and
the GitHub username in `dev_details` does not look similar to the author name in
`author_details`.

##### No Match -- Names Different, More Likely Match in Co-Contributors / Co-Authors

| dev_details 	| author_details 	| match 	|
|---	|---	|---	|
| username: mattwthompson;<br>name: Matt Thompson;<br>email: redacted@redacted.org;<br>repos: https://github.com/shirtsgroup/physical_validation;<br>co_contributors: ptmerz, dependabot[bot], mrshirts, pre-commit-ci[bot], wehs7661, cwalker7, MatKie, tlfobe, lgtm-com[bot]; 	| name: S. T. Boothroyd;<br>repos: https://github.com/shirtsgroup/physical_validation;<br>co_authors: Wei-tse Hsu, Michael R. Shirts, Chris C. Walker, Matthew W Thompson, Pascal Timothã©e Merz; 	| FALSE 	|

This row is labeled `False` because the provided name in `dev_details` is different
from the name provided in the `author_details`. Additionally, within the author details
we see a member of co-authors with the name `Matthew W Thompson` which seems like
a better match for the current `dev_details` under consideration.

##### Match -- Names Same, No More Likely Match in Co-Contributors / Co-Authors

| dev_details 	| author_details 	| match 	|
|---	|---	|---	
| username: sepandhaghighi;<br>name: Sepand Haghighi;<br>email: None;<br>repos: https://github.com/sepandhaghighi/pyrgg, https://github.com/sepandhaghighi/pycm, https://github.com/ECSIM/opem;<br>co_contributors: sadrasabouri, ivanovmg, ahmadsalimi, dependabot-preview[bot], dependabot[bot], codacy-badger, sadrasabouri, alirezazolanvari, pyup-bot, dependabot-preview[bot], dependabot[bot], GeetDsa, negarzabetian, robinsonkwame, lewiuberg, mahi97, MasoomehJasemi, soheeyang, cclauss, the-lay, pyup-bot, mahi97, sadrasabouri, dependabot-preview[bot], nnadeau, kasraaskari, engnadeau, dependabot[bot]; 	| name: Sepand Haghighi;<br>repos: https://github.com/sepandhaghighi/pyrgg, https://github.com/sepandhaghighi/pycm;<br>co_authors: Masoomeh Jasemi, Alireza Zolanvari, Shaahin Hessabi; 	| TRUE 	|

This row is labeled `True` because the provided name in `dev_details` is the same.
Additionally, within the author details we see no co-authors with a name that looks
like a better match for the current `dev_details` under consideration.

##### Match -- GitHub Username or Email Similar to Author Name, No More Likely Match in Co-Contributors / Co-Authors

| dev_details 	| author_details 	| match 	|
|---	|---	|---	|
| username: kdpeterson51;<br>name: None;<br>email: None;<br>repos: https://github.com/kdpeterson51/mbir;<br>co_contributors: arcaldwell49, iriberri;   | name: Kyle Bradley Peterson;<br>repos: https://github.com/kdpeterson51/mbir;<br>co_authors: Aaron Richard Caldwell;   | TRUE 	|

This row is labeled `True` because the provided GitHub username in `dev_details`
looks similar to the author name in `author_details`. Additionally, within the
author details we see no co-authors with a name that looks like a better match
for the current `dev_details` under consideration.

#### Results of IRR

TODO