import pandas as pd
import numpy as np  
import random
from sklearn.metrics.pairwise import cosine_similarity

print('Welcome to Dota Captain\'s draft')
side = input('Which side are you playing for?\n A. Radiant \tB. Dire\n')
ban_first = input('Do you wish to take first ban?\n A. Yes \tB. No\n')

drafts = pd.read_csv('./data/sparse_matrix.csv', index_col=0)
max_index = drafts.index.max()
new_index = max_index + 1
new_row = pd.DataFrame([[0] * drafts.shape[1]], columns=drafts.columns, index=[new_index])

ancient1 = 'radiant' if side == 'A' else 'dire'
ancient2 = 'radiant' if side == 'B' else 'dire'
drafts = pd.concat([drafts, new_row])

all_drafts = drafts.drop(columns=['radiant_win', 'team'])

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    return similarity

def rank_similarities(df):
    # Ensure the DataFrame is numerical
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    last_entry = df.iloc[-1].values
    other_entries = df.iloc[:-1]
    
    # Compute cosine similarities
    similarities = []
    for idx, row in other_entries.iterrows():
        similarity = cosine_similarity(last_entry, row.values)
        similarities.append((idx, similarity))
    
    # Create a DataFrame for the results with original indices
    similarity_df = pd.DataFrame(similarities, columns=['Index', 'Cosine_Similarity'])
    
    # Sort the results in descending order
    ranked_df = similarity_df.sort_values(by='Cosine_Similarity', ascending=False)
    
    return ranked_df



df_heroes = pd.read_csv('./data/Constants/Constants.Heroes.csv', usecols=['localized_name', 'id'], index_col='id')
def get_hero_name(id):
    return df_heroes.loc[id, 'localized_name']

def get_hero_id(name):
    # Convert the search name to lowercase
    name = name.lower()
    
    # Convert the 'localized_name' column to lowercase for comparison
    df_heroes_lower = df_heroes.copy()
    df_heroes_lower['localized_name'] = df_heroes_lower['localized_name'].str.lower()
    
    # Filter the DataFrame by the given name
    result = df_heroes_lower[df_heroes_lower['localized_name'] == name]
    
    if not result.empty:
        return result.index[0]  # Return the first matching index (id)
    else:
        return None  # Return None if no matching name is found
    
    
def print_similar(similar_drafts, order, current_draft):
    print('\t\t---Suggested Heroes---')
    top_n_similar = similar_drafts.sort_values(by='Cosine_Similarity', ascending=False)
    top_n_similar = top_n_similar.head(5)
    recommendations = []    
    for index in top_n_similar.index:
        # Get the row from the drafts DataFrame corresponding to the current index
        similar_row = all_drafts.iloc[index]
        
        # Filter columns in the row that match the pattern '_order'
        matching_columns = similar_row.filter(like=f'_{order}')
        
        # Find the matching columns that have a value of 1
        matching_column_ids = matching_columns[matching_columns == 1].index.tolist()
        
        # Print the matching values if any
        if matching_column_ids:
            cleaned_column_ids = [ids.split('_')[0] for ids in matching_column_ids]
            converted_names = [get_hero_name(int(id)) for id in cleaned_column_ids]
            new_recommendations = [name for name in converted_names if name not in current_draft]
            unique_recommendations = list(set(new_recommendations))
            recommendations.extend(unique_recommendations)        
    print(f'\t\t{list(set(recommendations))[:5]}')

first_phase_ban_1 = [1, 4, 7]
first_phase_ban_2 = [2, 3, 5, 6]

first_phase_pick_1 = [8]
first_phase_pick_2 = [9]

second_phase_ban_1 = [10, 11]
second_phase_ban_2 = [12]

second_phase_pick_1 = [14, 15, 18]
second_phase_pick_2 = [13, 16, 17]

third_phase_ban_1 = [19, 22]
third_phase_ban_2 = [20, 21]

third_phase_pick_1 = [23]
third_phase_pick_2 = [24]

first_team = first_phase_ban_1 + first_phase_pick_1 + second_phase_ban_1 + second_phase_pick_1 + third_phase_ban_1 + third_phase_pick_1
second_team = first_phase_ban_2 + first_phase_pick_2 + second_phase_ban_2 + second_phase_pick_2 + third_phase_ban_2 + third_phase_pick_2


all_bans = first_phase_ban_1 + first_phase_ban_2 + second_phase_ban_1 + second_phase_ban_2 + third_phase_ban_1 + third_phase_ban_2
optimized_ban_picks = [1, 2, 2, 3, 4, 4, 5, 6, 7, 8 , 8, 9, 10, 11, 11, 12, 12, 13 ,14, 15, 15, 16, 17, 18][::-1]

bans_one = []
bans_two = []
team_one = []
team_two = []
current_draft = []

# Dictionary to map index to phase
phase_mapping = {i: 'First' for i in range(1, 10)}
phase_mapping.update({i: 'Second' for i in range(10, 19)})
phase_mapping.update({i: 'Third' for i in range(19, 25)})


for i in range(1, 25):
    # Determine whether to ban or pick
    banpick = 'ban' if i in all_bans else 'pick'
    
    # Get the current phase
    phase = phase_mapping.get(i, '')
    
    # Pop the last element from optimized_ban_picks
    optimized_order = optimized_ban_picks.pop()
    
    # Determine which team is taking the action
    if i in first_team:
        team = 'TEAM 1'
        team_list = team_one if banpick == 'pick' else bans_one
        if ban_first == 'A' and i != 1:
            print_similar(similar_drafts, optimized_order, current_draft)
        
    elif i in second_team:
        team = 'TEAM 2'
        team_list = team_two if banpick == 'pick' else bans_two
        if ban_first == 'B':
            print_similar(similar_drafts, optimized_order, current_draft)
    else:
        continue
    prompt = '{} {} hero:'.format(team, banpick)
    # Get hero name from input
    name = input(prompt)
    # print(f'\t{phase} {banpick} phase: selected hero for {team}: {name}')
    
    # Append the hero to the appropriate team list if it's a pick
    team_list.append(name)
    # Add the hero to the bans list if it's a ban
    # if i in all_bans:
    #     bans.append(name)
    
    # Update the current draft and related data
    current_draft = bans_one + bans_two + team_one + team_two
    value = get_hero_id(name)
    column_name = f'{value}_{optimized_order}'
    all_drafts.iloc[-1, all_drafts.columns.get_loc(column_name)] = 1
    similar_drafts = rank_similarities(all_drafts)
    
    print(f'\n---{phase} PHASE {banpick}---')
    print(f'TEAM 1 BANS: {bans_one}')
    print(f'TEAM 1 PICKS: {team_one}\n')
    print(f'TEAM 2 BANS: {bans_two}')
    print(f'TEAM 2 PICKS: {team_two}')
    