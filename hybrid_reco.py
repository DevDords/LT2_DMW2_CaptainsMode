import numpy as np
import pandas as pd
import pickle
import os

# get hero name/id
def hero_id_name(key, get_what='name'):
    if get_what == 'name':
        return df_heroes.loc[df_heroes['id'] == key, 'localized_name'].iloc[0]
    else:
        return df_heroes.loc[df_heroes['localized_name'] == key, 'id'].iloc[0]


# return recommendation hero codes based on user based collaborative filtering
def cosim_recos(utility_matrix, phase, current_draft, top_n=20):
    sliced_matrix = slice_utility(utility_matrix, phase)
    draft_heroids = [hero_id_name(key, 'id') for key in current_draft]
    cols_to_get = [col for col in utility_matrix.columns if int(str(col).split('_')[-1]) == phase and int(str(col).split('_')[0]) not in draft_heroids]
    recos = []

    # calculate similarity matrix
    last_row = sliced_matrix.iloc[-1].to_numpy()
    matrix = sliced_matrix.to_numpy()
    dot_products = np.dot(matrix, last_row)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(last_row)
    similarities = np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms != 0)
    similarity_scores = pd.Series(similarities[:-1], index=sliced_matrix.index[:-1])
    match_id = similarity_scores.sort_values(ascending=False).index

    # check if the reco from highest similarity is already in current draft
    for i in match_id:
        reco = utility_matrix.loc[i, cols_to_get].eq(1)
        recos += reco[reco].index.tolist()
        if len(recos) >= top_n:
            break

    return [int(str(i).split('_')[0]) for i in recos]

def print_list(recos):
    for reco in recos:
        print(f'\t{reco}')

# run recommendations with assoc rules
def check_recos(is_ban, enemy_bans, enemy_picks, user_picks, recos, current_draft):
    low_consequents = []
    high_consequents = []
    enemy_bans = [hero_id_name(key, 'id') for key in enemy_bans]
    enemy_picks = [hero_id_name(key, 'id') for key in enemy_picks]
    user_picks = [hero_id_name(key, 'id') for key in user_picks]

    if is_ban == 'ban':
        for rule in low_rules: # get consequent with lowest lift based on enemy ban (their target hero)
            if rule['antecedent'] == tuple(enemy_bans) and rule['consequent'] in recos:
                low_consequents.append((
                    hero_id_name(rule['consequent']),
                    f"confidence: {rule['confidence']}",
                    f"lift: {rule['lift']}"
                    ))
            if len(low_consequents) == 3:
                break  # Stop after finding the top 3
        if enemy_picks:
            for rule in rules:  # get consequent with highest lift based on enemy picks (synergy picks)
                if rule['antecedent'] == tuple(enemy_picks) and rule['consequent'] in recos:
                    high_consequents.append((
                        hero_id_name(rule['consequent']),
                        f"confidence: {rule['confidence']}",
                        f"lift: {rule['lift']}"
                        ))
                if len(high_consequents) == 3:
                    break  # Stop after finding the top 3
        if len(low_consequents) + len(high_consequents) > 0:
            if len(low_consequents)  > 0:
                print(f"Enemy is trying to pick any of these heroes based on their ban:")
                print_list(low_consequents)
            if len(high_consequents) > 0:
                print(f"Enemy wants to combo their heroes with one of these heroes:")
                print_list(high_consequents)
            print(f"Recommended {is_ban}s based on pick and bans order:\n\t{[hero_id_name(i) for i in recos[:3]]}")
        else:
            print(f"Recommended {is_ban}s based on pick and bans order:\n\t{[hero_id_name(i) for i in recos[:3]]}")
    else:
        if not user_picks and not enemy_picks: # select first pick from most band or most first picked hero (META OR OP HERO)
            get_priority_heroes(sparse_matrix, current_draft)
        else:            
            for rule in low_rules: # get consequent with lowest lift based on enemy picks (probable counterpick)
                if rule['antecedent'] == tuple(enemy_picks) and rule['consequent'] in recos:
                    low_consequents.append((
                        hero_id_name(rule['consequent']),
                        f"confidence: {rule['confidence']}",
                        f"lift: {rule['lift']}"
                        ))
                if len(low_consequents) == 3:
                    break  # Stop after finding the top 3
            if user_picks:
                for rule in rules:  # get consequent with highest lift based on user picks (synergy picks)
                    if rule['antecedent'] == tuple(user_picks) and rule['consequent'] in recos:
                        high_consequents.append((
                            hero_id_name(rule['consequent']),
                            f"confidence: {rule['confidence']}",
                            f"lift: {rule['lift']}"
                            ))
                    if len(high_consequents) == 3:
                        break  # Stop after finding the top 3
            if len(low_consequents) + len(high_consequents) > 0:
                print(f"Probably counter to enemy picks:")
                print_list(low_consequents)
                print(f"Combo your hero with these heroes:")
                print_list(high_consequents)
                print(f"Recommended {is_ban}s based on pick and bans order:\n\t{[hero_id_name(i) for i in recos[:3]]}")
            else:
                print(f"Recommended {is_ban}s based on pick and bans order:\n\t{[hero_id_name(i) for i in recos[:3]]}")


# get user side of user
def get_side():
    while True:
        side = input('Which side are you playing for?\n A. Radiant \tB. Dire\n').strip().upper()
        if side in ['A', 'B']:
            side = 'radiant' if side == 'A' else 'dire'
            print("Your side is", side.title(), "\n")
            return side
        else:
            print("Invalid choice. Please enter 'A' for Radiant or 'B' for Dire.", "\n")


# determine team to first move
def get_ban_first():
    while True:
        ban_first = input('Do you wish to take first ban?\n Y = Yes \tN = No\n').strip().upper()
        if ban_first in ['Y', 'N']:
            user = 'TEAM 1 (first to ban)' if ban_first == 'Y' else 'TEAM 2 (last to pick)'            
            enemy = 'TEAM 1 (first to ban)' if ban_first == 'N' else 'TEAM 2 (last to pick)'
            print("You are", user)
            print("Enemy is", enemy, "\n")
            return ban_first
        else:
            print("Invalid choice. Please enter 'Y' for Yes or 'N' for No.", "\n")

# generate match id
def generate_match_id(recent_match_id):
    return recent_match_id.index.max() + 1


# insert match to db
def insert_match(match_db, match_id):
    new_row = pd.DataFrame([[0] * match_db.shape[1]], columns=match_db.columns, index=[match_id])
    
    return pd.concat([match_db, new_row])


# filter utility matrix to get recommendations aligned with user order of picks and bans
def filter_winning_side(match_db, ban_first):
    match_db['ban_first_win'] = (match_db['radiant_win'] & (match_db['team'] == 0)) | (~match_db['radiant_win'] & (match_db['team'] == 1))

    if ban_first == 'Y':
        filtered_db = match_db[match_db['ban_first_win']].copy()
    else:
        filtered_db = match_db[~match_db['ban_first_win']].copy()

    return filtered_db.drop(columns=['radiant_win', 'team', 'ban_first_win'])


# get priority first pick or ban
def get_priority_heroes(match_db, current_draft):
    draft_heroids = [hero_id_name(key, 'id') for key in current_draft]
    priority_bans = match_db.filter(regex='_1$').sum(axis=0)
    priority_picks = match_db.filter(regex='_8$').sum(axis=0)
    priority_picks = priority_picks.loc[~priority_picks.index.str.replace('_8', '').astype(int).isin(draft_heroids)]
    top_bans_id = priority_bans.nlargest(3).index.str.replace('_1', '').astype(int).tolist()
    top_picks_id = priority_picks.nlargest(3).index.str.replace('_8', '').astype(int).tolist()
    top_bans_name = [f"{hero_id_name(id, 'name')}" for id in top_bans_id]
    top_picks_name = [f"{hero_id_name(id, 'name')}" for id in top_picks_id]

    print(
        f"Most banned heroes that are still available: {', '.join(top_bans_name)}\n"
        f"Most picked heroes that are still available: {', '.join(top_picks_name)}", "\n"
    )


# slice utility matrix based on draft phase (context-aware cos sim)
def slice_utility(utility_matrix, phase):
    cols = [col for col in utility_matrix.columns if int(str(col).split('_')[-1]) < phase]
    return utility_matrix[cols]
    
def get_win_rate(only_drafts, is_team1=True):
    # Calculate dot products and norms for similarity scores
    last_row = only_drafts.iloc[-1].to_numpy()
    matrix = only_drafts.to_numpy()
    
    dot_products = np.dot(matrix, last_row)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(last_row)
    similarities = np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms != 0)
    
    # Get the similarity scores for all matches except the last one
    similarity_scores = pd.Series(similarities[:10], index=only_drafts.index[:10])
    
    # Use vectorized operations to calculate win count
    match_ids = similarity_scores.index
    radiant_wins = matches.loc[match_ids, 'radiant_win'].values
    teams_selected = matches.loc[match_ids, 'team'].values
    
    # Determine the win conditions based on the team
    if is_team1:
        win_conditions = (teams_selected == 0) & (radiant_wins == True) | (teams_selected == 1) & (radiant_wins == False)
    else:
        win_conditions = (teams_selected == 0) & (radiant_wins == False) | (teams_selected == 1) & (radiant_wins == True)
    
    win_count = np.sum(win_conditions)
    
    return win_count

def start_draft(utility_matrix, ban_first):

    first_team = [1, 4, 7, 8, 10, 11, 14, 15, 18, 19, 22, 23]  # first team order of ban/pick
    second_team = [2, 3, 5, 6, 9, 12, 13, 16, 17, 20, 21, 24]  # second team order of ban/pick
    phase_mapping = {i: ('First' if i < 10 else 'Second' if i < 19 else 'Third') for i in range(1, 25)}  # mapping for phase ban or pick
    all_bans = [1, 4, 7, 2, 3, 5, 6, 10, 11, 12, 19, 22, 20, 21]  # ban orders
    order_phase = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 8, 12: 9, 13: 10, 14: 11,
                   15: 11, 16: 12, 17: 12, 18: 13, 19: 14, 20: 15, 21: 15, 22: 16, 23: 17, 24: 18}  # dict for converting order to phase order
    subscript_value = list(order_phase.values())  # for updating specific columns based on phase

    # Initialize draft state
    bans_team1, bans_team2 = [], []
    picks_team1, picks_team2 = [], []
    current_draft = []
    
    for turn in range(1, 25):
        # Determine whether to ban or pick
        banpick = 'ban' if turn in all_bans else 'pick'
    
        # Pop the last element from subscript_value
        column_tag = subscript_value.pop()
    
        # Get the current phase
        phase = phase_mapping.get(turn, '')
        
        # Determine which team is taking the action
        if turn in first_team:
            team = 'TEAM 1'
            team_list = picks_team1 if banpick == 'pick' else bans_team1
            if ban_first == 'Y' and turn != 1:
                check_recos(banpick, bans_team2, picks_team2, picks_team1, index_recos, current_draft)
            elif ban_first == 'Y' and turn == 1:
                get_priority_heroes(sparse_matrix, current_draft)
        
        elif turn in second_team:
            team = 'TEAM 2'
            team_list = picks_team2 if banpick == 'pick' else bans_team2
            if ban_first == 'N':
                check_recos(banpick, bans_team1, picks_team1, picks_team2, index_recos, current_draft)
        else:
            continue

        # get input for ban or pick    
        while True:
            prompt = '{} {} hero: '.format(team, banpick)
            # Get hero name from input
            name = input(prompt)
            if name == 'exit':
                break
            if name in current_draft:
                print('Hero is already banned or picked')
                continue
            
            # Attempt to get hero ID
            try:
                value = hero_id_name(name, 'id')
                break  # Exit loop if no error
            except Exception as e:
                print(f"Error: Please enter a valid hero name.")

        team_list.append(name)
        
        # Update the current draft and related data
        current_draft = bans_team1 + bans_team2 + picks_team1 + picks_team2
        column_name = f'{value}_{column_tag}'
        utility_matrix.iloc[-1, utility_matrix.columns.get_loc(column_name)] = 1
        index_recos = cosim_recos(utility_matrix, order_phase[turn], current_draft, 20)
        
        print(f'\n---{phase} PHASE {banpick}---')
        print(f"User is TEAM {1 if ban_first == 'Y' else 2}")
        print(f'TEAM 1 BANS: {bans_team1}')
        print(f'TEAM 1 PICKS: {picks_team1}\n')
        print(f'TEAM 2 BANS: {bans_team2}')
        print(f'TEAM 2 PICKS: {picks_team2}\n\n')
    
    return picks_team1, picks_team2


if __name__ == "__main__":
    # load utility matrix
    sparse_matrix = pd.read_csv('./data/sparse_matrix.csv', index_col=0)

    # load heroes reference
    df_heroes = pd.read_csv('./data/Constants/Constants.Heroes.csv', usecols=['id', 'localized_name'])
    
    # load matches reference for winrate
    matches = pd.read_csv('./data/whole_draft.csv', index_col=0, low_memory=False)
    
    only_drafts = matches.drop(columns=['radiant_win', 'team'])
    
    # import assoc rules
    with open('./data/rules.pkl', 'rb') as file:
        rules = pickle.load(file)

    # lowest lifts (to prevent sorting everytime)
    low_rules = sorted(rules, key=lambda x: x['lift'])

    # get  match information
    draft_db = sparse_matrix.copy()
    side = get_side()
    ban_first = get_ban_first()

    # filter to get recs only from same side winning lineup with same draft order as user
    filtered_db = filter_winning_side(draft_db, ban_first)

    # insert current match to utility matrix
    match_id = generate_match_id(draft_db)
    draft_id = generate_match_id(only_drafts)

    # utility matrix for recommender
    utility_matrix = insert_match(filtered_db, match_id)
    only_drafts = insert_match(only_drafts, draft_id)
    
    picks_1, picks_2 = start_draft(utility_matrix, ban_first)
    
    user_team, enemy_team = (picks_1, picks_2) if ban_first == 'Y' else (picks_2, picks_1)
    
    
    # user_team = ['Shadow Fiend', 'Muerta']
    # enemy_team = ['Bristleback', 'Slark']
    
    # Extract column indices based on hero IDs
    user_ids = [hero_id_name(hero, 'id') for hero in user_team]
    enemy_ids = [hero_id_name(hero, 'id') for hero in enemy_team]

  
    for id in user_ids:
        only_drafts.loc[draft_id, id] = 1
    for id in enemy_ids:
        only_drafts.loc[draft_id, id] = 2

    # Call the function to get win rate
    sim1 = get_win_rate(only_drafts, True)
    
    for id in user_ids:
        only_drafts.loc[draft_id, id] = 2
    for id in enemy_ids:
        only_drafts.loc[draft_id, id] = 1

    sim2 = get_win_rate(only_drafts, False)
    
    print(f'Winning Probability: {(sim1+sim2)/20}')