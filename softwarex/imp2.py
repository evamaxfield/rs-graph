import requests
import pandas as pd


api_token = ''

org_url = "https://api.github.com/orgs/ElsevierSoftwareX/repos"

headers = {
    'Authorization': f'token {api_token}',
    'Accept': 'application/vnd.github.v3+json',
}

repos = []
while org_url:
    response = requests.get(org_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}")
    repos.extend(response.json())
        org_url = response.links.get('next', {}).get('url', None)

# fetch repository urls and parent urls (fork)
repo_data = []
for repo in repos:
    if repo['fork']:
        print(f"Processing forked repository: {repo['name']}")
        #print statements to debug errors
        repo_detail_response = requests.get(repo['url'], headers=headers)
        if repo_detail_response.status_code != 200:
            print(f"Failed to fetch details for {repo['name']}: {repo_detail_response.status_code}")
            continue  # Skip 
        repo_detail = repo_detail_response.json()
        try:
            parent_url = repo_detail['parent']['html_url']
            repo_data.append((repo['html_url'], parent_url))
        except KeyError:
            print(f"Could not retrieve parent info for {repo['name']}. Full repo details: {repo_detail}")
            continue  #iterate if needed
    else:
        print(f"Skipping non-forked repository: {repo['name']}")

# make a csv
df = pd.DataFrame(repo_data, columns=['repo_url', 'original_repo_url'])
df.to_csv('/forked_repos-origins.csv', index=False)
