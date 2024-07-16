import bittensor as bt
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.model_tracker import ModelTracker
import asyncio
from tqdm import tqdm
import requests
from datetime import datetime
from prettytable import PrettyTable

subtensor = bt.subtensor()
metagraph = subtensor.metagraph(11)
metadata_store = ChainModelMetadataStore(
            subtensor, 11, 'sn9'
        )
model_tracker = ModelTracker()
# URL of the JSON file
url = "http://34.41.206.211:8000/leaderboard"

# Fetching the JSON data
response = requests.get(url)
data = response.json()

model_infos = {}
for info in data:
    timestamp_str = info['timestamp']
    repo_namespace = info['repo_namespace']
    repo_name = info['repo_name']
    total_score = info['total_score']
    if "." in timestamp_str:
        date_part, microseconds_part = timestamp_str.split(".")
        microseconds_part = microseconds_part.rstrip("Z")
        if "+" in microseconds_part:
            microseconds_part, timezone_part = microseconds_part.split("+")
            timestamp_str = f"{date_part}.{microseconds_part.ljust(6, '0')}+{timezone_part}"
        else:
            timestamp_str = f"{date_part}.{microseconds_part.ljust(6, '0')}Z"
    timestamp_dt = datetime.fromisoformat(timestamp_str)
    unix_timestamp = timestamp_dt.timestamp()
    model_infos[(repo_namespace, repo_name)] = {'timestamp': unix_timestamp, 'total_score': total_score}#[unix_timestamp, total_score]

model_infos_filtered = {}
for hotkey in tqdm(metagraph.hotkeys):
    metadata = asyncio.run(
                        metadata_store.retrieve_model_metadata(hotkey)
                    )
    if metadata is None:
        continue
    if (metadata.id.namespace, metadata.id.name) in model_infos:
        model_infos[(metadata.id.namespace, metadata.id.name)]['block'] = metadata.block
        model_infos[(metadata.id.namespace, metadata.id.name)]['hotkey'] = hotkey
        model_infos_filtered[(metadata.id.namespace, metadata.id.name)] = model_infos[(metadata.id.namespace, metadata.id.name)]

model_wins = {}
for key1 in model_infos_filtered:
    model_wins[key1] = [0, model_infos_filtered[key1]['total_score'], model_infos_filtered[key1]['block'], model_infos_filtered[key1]['hotkey']]
    score1 = model_infos_filtered[key1]['total_score']
    block1 = model_infos_filtered[key1]['block']
    for key2 in model_infos_filtered:
        if key1 == key2:
            continue
        score2 = model_infos_filtered[key2]['total_score']
        block2 = model_infos_filtered[key2]['block']
        if block1 > block2:
            score1_adj = score1 * 0.975
            score2_adj = score2
        else:
            score2_adj = score2 * 0.975
            score1_adj = score1
        if score1_adj > score2_adj:
            model_wins[key1][0] += 1

model_wins = dict(sorted(model_wins.items(), key=lambda item: item[1], reverse=True))

table = PrettyTable()
table.field_names = ["Namespace", "Name", "Wins", "Total Score", "Block", "Hotkey"]
for key in model_wins.keys():
    table.add_row([key[0], key[1], model_wins[key][0], model_wins[key][1], model_wins[key][2], model_wins[key][3]])
#print(model_wins)

print(table)
