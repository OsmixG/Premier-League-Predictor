import pandas as pd
import numpy as np
import time

"""
#####################################################################################################
For the prediction of the match that is not played yet, the statistics of that match cannot be used.
Therefore, we planned to modelize our input vector X from the statistics of previous matches. For example,
we cannot plug in the shot on target of the teams from the non-played match, instead, we model three main features for each of the teams:
(In total we 23x2+ 1 = 47 features for each match)
Each team have different dataset for home and away matches. For example, Chelsea has two dataset: one for home matches and one for away matches.
1) Attack srength:
    - gf_last15_avg: It is calculated by the total goals scored by the team in the last 15 matches before that match.
    - gf_last5_avg: It is calculated by the total goals scored by the team in the last 5 matches before that match.
    - shot_on_target_last15_avg: It is calculated by the total shots on target of the team in the last 15 matches before that match.
    - shot_on_target_last5_avg: It is calculated by the total shots on target of the team in the last 5 matches before that match.
    - shot_last15_avg: It is calculated by the total shots of the team in the last 15 matches before that match.
    - shot_last5_avg: It is calculated by the total shots of the team in the last 5 matches before that match.
    - corner_last15_avg: It is calculated by the total corners of the team in the last 15 matches before that match.
    - corner_last5_avg: It is calculated by the total corners of the team in the last 5 matches before that match.
    - faul_againts_last15_avg: It is calculated by the total fauls of the OPPENENTS in the last 15 matches before that match.
    - faul_againts_last5_avg: It is calculated by the total fauls of the OPPENENTS in the last 5 matches before that match.

2) Defence strength:

    - ga_last5_avg: It is calculated by the total goals conceded by the team in the last 5 matches before that match.
    - zga_last5_avg: It is calculated by the total goals conceded by the team in the last 15 matches before that match.
    - ga_last15_avg: It is calculated by the total goals conceded by the team in the last 15 matches before that match.
    - shot_on_target_against_last5_avg: It is calculated by the total shots on target against the team in the last 5 matches before that match.
    - shot_on_target_against_last15_avg: It is calculated by the total shots on target against the team in the last 15 matches before that match.
    - shot_against_last5_avg: It is calculated by the total shots against the team in the last 5 matches before that match.
    - shot_against_last15_avg: It is calculated by the total shots against the team in the last 15 matches before that match.
    - faul_last5_avg: It is calculated by the total fauls of the team in the last 5 matches before that match.
    - faul_last15_avg: It is calculated by the total fauls of the team in the last 15 matches before that match.
3) Discipline:
    - card_points_last15_avg: It is calculated by the total card points of the team in the last 15 matches before that match.
    - card_points_last5_avg: It is calculated by the total card points of the team in the last 5 matches before that match.
    - cord_points_per_faul_last15_avg: It is calculated by the total card points per faul of the team in the last 15 matches before that match.
    - cord_points_per_faul_last5_avg: It is calculated by the total card points per faul of the team in the last 5 matches before that match.

The last X vecotor = [1, Away_team_attack_strength, Away_team_defence_strength, Away_team_discipline, Home_team_attack_strength, 
Home_team_defence_strength, Home_team_discipline]

Data leakage:
In the process of creating the input vector X, we have to be careful about data beacuse some season some of the set of the teams
cannot be played due to bad result of the previous reason. Therefore, some teams played less match than others. While taking the
average of the last 5 matches, some of the teams haven't played the match in PL for a few seasons from the day of the match.
Therefore, we arranged the data and put zero for the matches that they haven't played. For example, if a team has played only 3 matches
before the match that we want to predict, we take the average of those 3 matches and put zero for the rest of the 2 matches. 
In this way, we can avoid data leakage and make our model more robust.

Independency: 
In the theoretical world, the input vectors dimensions are assumed to be independent. However, in the real world, there are some correlations 
between the features. For example, the attack strength and defence strength of a team are correlated. Therefore, we have to be
careful about the correlations between the features and try to minimize them as much as possible. Lasso?


######################################################################################################
"""


class PremierLeagueDataProcessor:
    def __init__(self, start = True):
        
        self.data_array = []
        self.data_array_standings_last = [[],[],[],[],[],[],[],[],[],[]]
        self.data_array_standings = []
        self.score = []
        self.teams = []
        self.teams_rates_all = []
        self.teams_statistics = []
        self.teams_away_statistics_pure = []
        self.teams_home_statistics_pure = []
        self.yellow_cards = []
        self.gecici = []
        self.arranged_data_home = []
        self.arranged_data_away = []
        self.max_games = 0
        self.data_frames = []
        self.all_matches_df = None
        self.dataset_df = None
        self.X = None
        self.y_raw = None
        self.y_int = None
        self.Y_one_hot = None
        self.cross_validation = 8
        self.X_train_df = None
        self.X_test_df =None
        self.y_train_df = None
        self.y_test_df =None
        self.Y_zero_vs_rest = []
        self.Y_one_vs_rest =[]
        self.Y_two_vs_rest = []
        self.beta = []
        self.X_train = [0]*self.cross_validation
        self.X_test = [0]*self.cross_validation
        self.y_train = [0]*self.cross_validation
        self.y_test = [0]*self.cross_validation
        
        self.start = start
        self.base_feature_names = [
            'gf_last15_avg', 'gf_last5_avg',
            'shot_on_target_last15_avg', 'shot_on_target_last5_avg',
            'shot_last15_avg', 'shot_last5_avg',
            'corner_last15_avg', 'corner_last5_avg',
            'faul_againts_last15_avg', 'faul_againts_last5_avg',
            'ga_last5_avg', 'zga_last5_avg', 'ga_last15_avg',
            'shot_on_target_against_last5_avg', 'shot_on_target_against_last15_avg',
            'shot_against_last5_avg', 'shot_against_last15_avg',
            'faul_last5_avg', 'faul_last15_avg',
            'card_points_last15_avg', 'card_points_last5_avg',
            'cord_points_per_faul_last15_avg', 'cord_points_per_faul_last5_avg'
        ]
        self.home_feature_names = [f'home_{name}' for name in self.base_feature_names]
        self.away_feature_names = [f'away_{name}' for name in self.base_feature_names]
        self.feature_names = self.home_feature_names + self.away_feature_names
        if self.start == True:
            self.run_all()
    def _parse_date_series(self, s):
        out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
        mask = out.isna()
        if mask.any():
            out2 = pd.to_datetime(s[mask], format="%d/%m/%y", errors="coerce")
            out.loc[mask] = out2
        return out

    def _find_entry(self, collection, team):
        for item in collection:
            if item['Team'] == team:
                return item
        raise ValueError(f"Team not found: {team}")

    def _resolve_team(self, team_name):
        normalized = team_name.strip().lower()
        for team in self.teams:
            if team.lower() == normalized:
                return team
        raise ValueError(f"Unknown team: {team_name}")

    def _one_hot(self, y, num_classes=3):
        out = np.zeros((len(y), num_classes), dtype=float)
        for idx, label in enumerate(y):
            out[idx, int(label)] = 1.0
        return out

    def _label_to_int(self, ftr, fthg, ftag):
        if isinstance(ftr, str):
            ftr = ftr.strip().upper()
            if ftr == 'H':
                return 0
            if ftr == 'D':
                return 1
            if ftr == 'A':
                return 2
        if float(fthg) > float(ftag):
            return 0
        if float(fthg) == float(ftag):
            return 1
        return 2

    def load_data(self):
        
        fifteen_sixteen = pd.read_csv('season-1516.csv')
        sixteen_seventeen = pd.read_csv('season-1617.csv')
        seventeen_eighteen = pd.read_csv('season-1718.csv')
        eighteen_nineteen = pd.read_csv('season-1819.csv')
        nineteen_twenty = pd.read_csv('season-1920.csv')
        twenty_twentyone = pd.read_csv('season-2021.csv')
        twentyone_twentytwo = pd.read_csv('season-2122.csv')
        twentytwo_twentythree = pd.read_csv('season-2223.csv')
        twentythree_twentyfour = pd.read_csv('season-2324.csv')
        twentyfour_twentyfive = pd.read_csv('season-2425.csv')

        self.data_array.append(fifteen_sixteen.to_numpy())
        self.data_array.append(sixteen_seventeen.to_numpy())
        self.data_array.append(seventeen_eighteen.to_numpy())
        self.data_array.append(eighteen_nineteen.to_numpy())
        self.data_array.append(nineteen_twenty.to_numpy())
        self.data_array.append(twenty_twentyone.to_numpy())
        self.data_array.append(twentyone_twentytwo.to_numpy())
        self.data_array.append(twentytwo_twentythree.to_numpy())
        self.data_array.append(twentythree_twentyfour.to_numpy())
        self.data_array.append(twentyfour_twentyfive.to_numpy())

        self.data_frames = [
            fifteen_sixteen,
            sixteen_seventeen,
            seventeen_eighteen,
            eighteen_nineteen,
            nineteen_twenty,
            twenty_twentyone,
            twentyone_twentytwo,
            twentytwo_twentythree,
            twentythree_twentyfour,
            twentyfour_twentyfive
        ]

        self.data_array = np.array(self.data_array, dtype=object)

        fifteen_sixteen_standings = pd.read_csv('1516_standings.csv')
        sixteen_seventeen_standings = pd.read_csv('1617_standings.csv')
        seventeen_eighteen_standings = pd.read_csv('1718_standings.csv')
        eighteen_nineteen_standings = pd.read_csv('1819_standings.csv')
        nineteen_twenty_standings = pd.read_csv('1920_standings.csv')
        twenty_twentyone_standings = pd.read_csv('2021_standings.csv')
        twentyone_twentytwo_standings = pd.read_csv('2122_standings.csv')
        twentytwo_twentythree_standings = pd.read_csv('2223_standings.csv')
        twentythree_twentyfour_standings = pd.read_csv('2324_standings.csv')
        twentyfour_twentyfive_standings = pd.read_csv('2425_standings.csv')

        for i in range(0,10):
            if i == 0:
                a = fifteen_sixteen_standings.to_numpy()
                self.data_array_standings.append(a[1:])
            elif i == 1:
                a = sixteen_seventeen_standings.to_numpy()
                self.data_array_standings.append(a[1:])
            elif i == 2:
                a = seventeen_eighteen_standings.to_numpy()
                self.data_array_standings.append(a[1:])
            elif i == 3:
                a = eighteen_nineteen_standings.to_numpy()
                self.data_array_standings.append(a[1:])
            elif i == 4:
                a = nineteen_twenty_standings.to_numpy()
                self.data_array_standings.append(a[1:])
            elif i == 5:
                a = twenty_twentyone_standings.to_numpy()
                self.data_array_standings.append(a[1:])
            elif i == 6:
                a = twentyone_twentytwo_standings.to_numpy()
                self.data_array_standings.append(a[1:])
            elif i == 7:
                a = twentytwo_twentythree_standings.to_numpy()
                self.data_array_standings.append(a[1:])
            elif i == 8:
                a = twentythree_twentyfour_standings.to_numpy()
                self.data_array_standings.append(a[1:])
            else:
                a = twentyfour_twentyfive_standings.to_numpy()
                self.data_array_standings.append(a[1:])

        self.data_array_standings = np.array(self.data_array_standings, dtype=object)

        required_cols = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
            'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF',
            'HC', 'AC', 'HY', 'AY', 'HR', 'AR'
        ]
        merged = []
        for season_idx, df in enumerate(self.data_frames):
            xdf = df[required_cols].copy()
            xdf['season_idx'] = season_idx
            xdf['row_idx'] = np.arange(len(xdf))
            xdf['parsed_date'] = self._parse_date_series(xdf['Date'])
            merged.append(xdf)

        self.all_matches_df = pd.concat(merged, ignore_index=True)
        self.all_matches_df['parsed_date_isna'] = self.all_matches_df['parsed_date'].isna().astype(int)
        self.all_matches_df = self.all_matches_df.sort_values(
            by=['parsed_date_isna', 'parsed_date', 'season_idx', 'row_idx'],
            kind='mergesort'
        ).reset_index(drop=True)
        self.all_matches_df['home_side_season_idx'] = self.all_matches_df.groupby(['season_idx', 'HomeTeam']).cumcount()
        self.all_matches_df['away_side_season_idx'] = self.all_matches_df.groupby(['season_idx', 'AwayTeam']).cumcount()

    def calculate_scores_and_rates(self):
        score_fifteen_sixteen = 0

        for j in range(len(self.data_array)):
            x = 0
            for i in range(len(self.data_array[j])):
                if self.data_array[j][i][5] == 'H':
                    x += 3
                elif self.data_array[j][i][5] == 'A':
                    x += 3
                else:
                    x += 1
            self.score.append(x)

        self.score = np.array(self.score)

        Total_score = 0
        for i in range(len(self.score)):
            Total_score += self.score[i]

        """Home rate and Away rate may be evelauted using just their types"""
        for j in range(len(self.data_array)):
            for i in range(len(self.data_array_standings[j])):
                x = {
                    'Team': self.data_array_standings[j][i][1],
                    'Score': int(self.data_array_standings[j][i][17]),
                    'Total_rate': int(self.data_array_standings[j][i][17]) / self.score[j],
                    'Home_rate': (int(self.data_array_standings[j][i][3]) * 3 + int(self.data_array_standings[j][i][4]) * 1 )/ self.score[j],
                    'Away_rate': (int(self.data_array_standings[j][i][7]) * 3 + int(self.data_array_standings[j][i][8]) * 1)/ self.score[j]
                }
                self.data_array_standings_last[j].append(x)

        self.data_array_standings_last = np.array([np.array(season) for season in self.data_array_standings_last], dtype=object)

        for data in self.data_array_standings_last:
            for i in data:
                team = i['Team']
                team = team.split(' (')[0]
                self.teams.append(team)

        self.teams = np.array(list(set(self.teams)))
            
        for i in self.teams:
            score_temp = 0 
            total_rate = 0
            home_rate = 0
            away_rate = 0
            for j in range(len(self.data_array_standings_last)):
                for k in range(len(self.data_array_standings_last[j])):
                    if self.data_array_standings_last[j][k]['Team'] == i:
                        total_rate += self.data_array_standings_last[j][k]['Total_rate']
                        home_rate += self.data_array_standings_last[j][k]['Home_rate']
                        away_rate += self.data_array_standings_last[j][k]['Away_rate']
            x = {'Team': i,
                 'Total_rate_all': total_rate,
                 'Home_rate_all': home_rate,
                 'Away_rate_all': away_rate}
            self.teams_rates_all.append(x)

        self.teams_rates_all = np.array(self.teams_rates_all, dtype=object)

    def process_match_statistics(self):
        #------------------------------------------------------
        """We classified away team and home team separetly. For example, Chealse has two folder: away and home"""
        for team in self.teams:
            home_statistics = []
            away_statistics = []
            for season in range(len(self.data_array)):
                season_columns = self.data_frames[season].columns
                for match in self.data_array[season]:    
                    row = dict(zip(season_columns, match))
                    if team == row['HomeTeam']:
                        a = {'FTHG': row['FTHG'],'HTHG': row['HTHG'], 'HS': row['HS'], 'HST': row['HST'], 'HF': row['HF'],'HC': row['HC'],'HY': row['HY'],'HR': row['HR'], 'GA': row['FTAG'], 'HTGA': row['HTAG'], 'FA': row['AF'], 'SA': row['AS'], 'STA': row['AST']}    
                        home_statistics.append(a)
                    elif team == row['AwayTeam']:
                        a = {'FTAG': row['FTAG'],'HTAG': row['HTAG'], 'AS': row['AS'], 'AST': row['AST'], 'AF': row['AF'],'AC': row['AC'],'AY': row['AY'],'AR': row['AR'], 'GA': row['FTHG'], 'HTGA': row['HTHG'], 'FA': row['HF'], 'SA': row['HS'], 'STA': row['HST']}    
                        away_statistics.append(a)
            
            x = {'Team': team, 'Statistics_away': np.array(away_statistics), 'Statistics_home': np.array(home_statistics)}
            self.teams_statistics.append(x)

        self.teams_statistics = np.array(self.teams_statistics, dtype=object)

        for team in self.teams:
            FTGAFH = []
            HTGAFH = []
            FTGAFA = []
            HTGAFA = []
            FTHG = []
            FTAG = []
            HTHG = []
            HTAG = []
            HS = []
            AS = []
            HST = []
            AST = []
            HF = []
            AF = []
            HC = []
            AC = []
            HY = []
            AY = []
            AR = []
            HR = []
            FAA, SAA, STAA = [], [], []
            FAH, SAH, STAH = [], [], []
            for i in self.teams_statistics:
                if i['Team'] == team:
                    for j in i['Statistics_away']:
                        FTAG.append(j['FTAG'])
                        HTAG.append(j['HTAG'])
                        AS.append(j['AS'])
                        AST.append(j['AST'])
                        AF.append(j['AF'])
                        AC.append(j['AC'])
                        AR.append(j['AR'])             
                        AY.append(j['AY'])
                        FTGAFA.append(j['GA'])
                        HTGAFA.append(j['HTGA'])
                        FAA.append(j['FA'])
                        SAA.append(j['SA'])
                        STAA.append(j['STA'])
                    for k in i['Statistics_home']:
                        FTHG.append(k['FTHG'])
                        HTHG.append(k['HTHG'])
                        HS.append(k['HS'])
                        HST.append(k['HST'])
                        HF.append(k['HF'])
                        HC.append(k['HC'])
                        HY.append(k['HY'])             
                        HR.append(k['HR'])
                        FTGAFH.append(k['GA'])
                        HTGAFH.append(k['HTGA'])
                        FAH.append(k['FA'])
                        SAH.append(k['SA'])
                        STAH.append(k['STA'])
            
            x = {'Team': team, 
                 'FTAG': np.array(FTAG), 'HTAG': np.array(HTAG), 'AS': np.array(AS), 
                 'AST': np.array(AST), 'AF': np.array(AF), 'AC': np.array(AC), 
                 'AR': np.array(AR), 'AY': np.array(AY), 'GA': np.array(FTGAFA), 'HTGA': np.array(HTGAFA), 'FA': np.array(FAA), 'SA': np.array(SAA), 'STA': np.array(STAA) }
            
            y = {'Team': team, 
                 'FTHG': np.array(FTHG), 'HTHG': np.array(HTHG), 'HS': np.array(HS), 
                 'HST': np.array(HST), 'HF': np.array(HF), 'HC': np.array(HC), 
                 'HY': np.array(HY), 'HR': np.array(HR), 'GA': np.array(FTGAFH), 'HTGA': np.array(HTGAFH),'FA': np.array(FAH), 'SA': np.array(SAH), 'STA': np.array(STAH)}
            
            self.teams_away_statistics_pure.append(x)
            self.teams_home_statistics_pure.append(y)

        self.teams_away_statistics_pure = np.array(self.teams_away_statistics_pure, dtype=object)
        self.teams_home_statistics_pure = np.array(self.teams_home_statistics_pure, dtype=object)

        for team in self.teams:
            for i in range(len(self.teams_away_statistics_pure)):
                if self.teams_away_statistics_pure[i]['Team'] == team:
                    if len(self.teams_away_statistics_pure[i]['AS']) > self.max_games:
                        self.max_games = len(self.teams_away_statistics_pure[i]['AS'])

    def handle_missing_seasons(self):
        for team in self.teams:
            season_number_array = []
            
            for season_number in range(len(self.data_array_standings)):
                it_is_not = 0
                for i in self.data_array_standings[season_number]:
                    if team != i[1]:
                        it_is_not = 1
                    else:
                        it_is_not = 0
                        break
                if it_is_not == 1:
                    season_number_array.append(season_number)
                else:
                    pass

            self.gecici.append({'Team': team, 'Season number': np.array(season_number_array)})    

        self.gecici = np.array(self.gecici, dtype=object)

        zero = np.zeros(19, dtype=int)

        for team in self.teams:
            for i in self.teams_away_statistics_pure:
                if i['Team'] == team:
                    for gecicix in self.gecici:
                        if gecicix['Team'] == team:
                            if len(gecicix['Season number']) > 0:
                                for j in range(len(gecicix['Season number'])):
                                    if  gecicix['Season number'][j] != 9:
                                        i['AS'] = np.insert(i['AS'], gecicix['Season number'][j]*19, zero)
                                        i['AST'] = np.insert(i['AST'], gecicix['Season number'][j]*19, zero)
                                        i['AF'] = np.insert(i['AF'], gecicix['Season number'][j]*19, zero)
                                        i['AC'] = np.insert(i['AC'], gecicix['Season number'][j]*19, zero)
                                        i['AR'] = np.insert(i['AR'], gecicix['Season number'][j]*19, zero)
                                        i['AY'] = np.insert(i['AY'], gecicix['Season number'][j]*19, zero)
                                        i['GA'] = np.insert(i['GA'], gecicix['Season number'][j]*19, zero)
                                        i['HTGA'] = np.insert(i['HTGA'], gecicix['Season number'][j]*19, zero)
                                        i['FTAG'] = np.insert(i['FTAG'], gecicix['Season number'][j]*19, zero)
                                        i['HTAG'] = np.insert(i['HTAG'], gecicix['Season number'][j]*19, zero)
                                        i['FA'] = np.insert(i['FA'], gecicix['Season number'][j]*19, zero)
                                        i['SA'] = np.insert(i['SA'], gecicix['Season number'][j]*19, zero)
                                        i['STA'] = np.insert(i['STA'], gecicix['Season number'][j]*19, zero)
                                        
                                    elif  gecicix['Season number'][j] == 9:
                                        i['AS'] = np.append(i['AS'], zero)
                                        i['AST'] = np.append(i['AST'], zero)
                                        i['AF'] = np.append(i['AF'], zero)
                                        i['AC'] = np.append(i['AC'], zero)
                                        i['AR'] = np.append(i['AR'], zero)
                                        i['AY'] = np.append(i['AY'], zero)
                                        i['GA'] = np.append(i['GA'], zero)
                                        i['HTGA'] = np.append(i['HTGA'], zero)
                                        i['FTAG'] = np.append(i['FTAG'], zero)
                                        i['HTAG'] = np.append(i['HTAG'], zero)
                                        i['FA'] = np.append(i['FA'], zero)
                                        i['SA'] = np.append(i['SA'], zero)
                                        i['STA'] = np.append(i['STA'], zero)

            for i in self.teams_home_statistics_pure:
                if i['Team'] == team:
                    for gecicix in self.gecici:
                        if gecicix['Team'] == team:
                            if len(gecicix['Season number']) > 0:
                                for j in range(len(gecicix['Season number'])):
                                    if  gecicix['Season number'][j] != 9:
                                        i['HS'] = np.insert(i['HS'], gecicix['Season number'][j]*19, zero)
                                        i['HST'] = np.insert(i['HST'], gecicix['Season number'][j]*19, zero)
                                        i['HF'] = np.insert(i['HF'], gecicix['Season number'][j]*19, zero)
                                        i['HC'] = np.insert(i['HC'], gecicix['Season number'][j]*19, zero)
                                        i['HR'] = np.insert(i['HR'], gecicix['Season number'][j]*19, zero)
                                        i['HY'] = np.insert(i['HY'], gecicix['Season number'][j]*19, zero)
                                        i['GA'] = np.insert(i['GA'], gecicix['Season number'][j]*19, zero)
                                        i['HTGA'] = np.insert(i['HTGA'], gecicix['Season number'][j]*19, zero)
                                        i['FTHG'] = np.insert(i['FTHG'], gecicix['Season number'][j]*19, zero)
                                        i['HTHG'] = np.insert(i['HTHG'], gecicix['Season number'][j]*19, zero)
                                        i['FA'] = np.insert(i['FA'], gecicix['Season number'][j]*19, zero)
                                        i['SA'] = np.insert(i['SA'], gecicix['Season number'][j]*19, zero)
                                        i['STA'] = np.insert(i['STA'], gecicix['Season number'][j]*19, zero)
                                        
                                    elif  gecicix['Season number'][j] == 9:
                                        i['HS'] = np.append(i['HS'], zero)
                                        i['HST'] = np.append(i['HST'], zero)
                                        i['HF'] = np.append(i['HF'], zero)
                                        i['HC'] = np.append(i['HC'], zero)
                                        i['HR'] = np.append(i['HR'], zero)
                                        i['HY'] = np.append(i['HY'], zero)
                                        i['GA'] = np.append(i['GA'], zero)
                                        i['HTGA'] = np.append(i['HTGA'], zero)
                                        i['FTHG'] = np.append(i['FTHG'], zero)
                                        i['HTHG'] = np.append(i['HTHG'], zero)
                                        i['FA'] = np.append(i['FA'], zero)
                                        i['SA'] = np.append(i['SA'], zero)
                                        i['STA'] = np.append(i['STA'], zero)
    def check_array_sizes(self):
        print("\n--- Home Statistics Array Sizes Check ---")
        for team_data in self.teams_home_statistics_pure:
            team_name = team_data['Team']
            print(f"\nTeam: {team_name}")
            
            for key, value in team_data.items():
                # 'Team' anahtarı bir string olduğu için onu atlıyoruz, sadece array'lere bakıyoruz
                if isinstance(value, (np.ndarray, list)):
                    print(f"  {key}: {len(value)}")
                else:
                    print(f"  {key}: (Not an array/list: {type(value).__name__})")
    def _history_window(self, arr, end_idx, window):
        hist = np.asarray(arr[max(0, end_idx - window):end_idx], dtype=float)
        if len(hist) < window:
            hist = np.pad(hist, (window - len(hist), 0), constant_values=0.0)
        return hist

    def _feature_row_from_side_stats(self, side_stats, side, end_idx):
        if side == 'away':
            gf = side_stats['FTAG']
            sot = side_stats['AST']
            shot = side_stats['AS']
            corner = side_stats['AC']
            foul_against = side_stats['FA']
            ga = side_stats['GA']
            sota = side_stats['STA']
            sa = side_stats['SA']
            foul = side_stats['AF']
            card_points = side_stats['AY'] + 2.0 * side_stats['AR']
        else:
            gf = side_stats['FTHG']
            sot = side_stats['HST']
            shot = side_stats['HS']
            corner = side_stats['HC']
            foul_against = side_stats['FA']
            ga = side_stats['GA']
            sota = side_stats['STA']
            sa = side_stats['SA']
            foul = side_stats['HF']
            card_points = side_stats['HY'] + 2.0 * side_stats['HR']

        def m(arr, window):
            return float(self._history_window(arr, end_idx, window).mean())

        def z(arr, window):
            return float((self._history_window(arr, end_idx, window) == 0).sum() / window)

        def r(num_arr, den_arr, window, eps=1e-8):
            num = self._history_window(num_arr, end_idx, window)
            den = self._history_window(den_arr, end_idx, window)
            return float(num.sum() / (den.sum() + eps))

        return np.array([
            m(gf, 15), m(gf, 5),
            m(sot, 15), m(sot, 5),
            m(shot, 15), m(shot, 5),
            m(corner, 15), m(corner, 5),
            m(foul_against, 15), m(foul_against, 5),
            m(ga, 5), z(ga, 5), m(ga, 15),
            m(sota, 5), m(sota, 15),
            m(sa, 5), m(sa, 15),
            m(foul, 5), m(foul, 15),
            m(card_points, 15), m(card_points, 5),
            r(card_points, foul, 15), r(card_points, foul, 5)
        ], dtype=float)

    def arrange_data(self):
        self.arranged_data_home = []
        self.arranged_data_away = []
        keys = ['GF15', 'GF5', 'SOT15', 'SOT5', 'Shot15', 'Shot5', 'Corner15', 'Corner5', 'FA15', 'FA5', 'GA5', 'Cleansheet', 'GA15', 'SOTA5', 'SOTA15', 'SA5', 'SA15', 'FF5', 'FF15', 'Card15', 'Card5', 'CPF15', 'CPF5']

        for team in self.teams:
            for source, side, target in [
                (self.teams_away_statistics_pure, 'away', self.arranged_data_away),
                (self.teams_home_statistics_pure, 'home', self.arranged_data_home)
            ]:
                for item in source:
                    if item['Team'] == team:
                        match_count = len(item['AS']) if side == 'away' else len(item['HS'])
                        series = {key: [] for key in keys}
                        feature_matrix = []
                        for j in range(match_count):
                            row = self._feature_row_from_side_stats(item, side, j)
                            feature_matrix.append(row)
                            for idx, key in enumerate(keys):
                                series[key].append(float(row[idx]))
                        target.append({
                            'Team': team,
                            'feature_matrix': np.array(feature_matrix, dtype=float),
                            'next_feature_vector': self._feature_row_from_side_stats(item, side, match_count),
                            'GF15': series['GF15'], 'GF5': series['GF5'],
                            'SOT15': series['SOT15'], 'SOT5': series['SOT5'],
                            'Shot15': series['Shot15'], 'Shot5': series['Shot5'],
                            'Corner15': series['Corner15'], 'Corner5': series['Corner5'],
                            'FA15': series['FA15'], 'FA5': series['FA5'],
                            'GA15': series['GA15'], 'GA5': series['GA5'],
                            'Cleansheet': series['Cleansheet'],
                            'SOTA15': series['SOTA15'], 'SOTA5': series['SOTA5'],
                            'SA15': series['SA15'], 'SA5': series['SA5'],
                            'FF15': series['FF15'], 'FF5': series['FF5'],
                            'Card15': series['Card15'], 'Card5': series['Card5'],
                            'CPF15': series['CPF15'], 'CPF5': series['CPF5']
                        })
                        break

        self.arranged_data_home = np.array(self.arranged_data_home, dtype=object)
        self.arranged_data_away = np.array(self.arranged_data_away, dtype=object)

    def build_final_x_matrix(self):
        rows = []
        X_list = []
        y_list = []

        for sample_id, (_, row) in enumerate(self.all_matches_df.iterrows()):
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            home_idx = int(row['season_idx']) * 19 + int(row['home_side_season_idx'])
            away_idx = int(row['season_idx']) * 19 + int(row['away_side_season_idx'])

            home_entry = self._find_entry(self.arranged_data_home, home_team)
            away_entry = self._find_entry(self.arranged_data_away, away_team)

            a_home_features = home_entry['feature_matrix'][home_idx].reshape(1, -1)
            b_away_features = away_entry['feature_matrix'][away_idx].reshape(1, -1)
            # Add bias term x_0 = 1 as the first column
            final_x_vector = np.concatenate([np.ones((1, 1)), a_home_features, b_away_features], axis=1)
            y_int = self._label_to_int(row['FTR'], row['FTHG'], row['FTAG'])

            item = {
                'sample_id': sample_id,
                'season_idx': int(row['season_idx']),
                'home_side_season_idx': int(row['home_side_season_idx']),
                'away_side_season_idx': int(row['away_side_season_idx']),
                'Date': row['Date'], 'HomeTeam': home_team, 'AwayTeam': away_team,
                'FTHG': int(row['FTHG']), 'FTAG': int(row['FTAG']), 'FTR': row['FTR'],
                'y_int': y_int, 'a_home_features': a_home_features,
                'b_away_features': b_away_features, 'x_vector': final_x_vector
            }
            for idx, name in enumerate(self.home_feature_names):
                item[name] = float(a_home_features[0, idx])
            for idx, name in enumerate(self.away_feature_names):
                item[name] = float(b_away_features[0, idx])

            rows.append(item)
            X_list.append(final_x_vector[0])
            y_list.append(y_int)

        self.dataset_df = pd.DataFrame(rows)
        self.X = np.array(X_list, dtype=float)

        mask = np.ones(len(self.X), dtype=bool)

        for i in range(len(self.X)):          
            if self.X[i, 5] == 0 or self.X[i, 28] == 0:
                mask[i] = False




        

        self.y_raw = self.dataset_df['FTR'].to_numpy()
        self.y_int = np.array(y_list, dtype=int)
        self.Y_one_hot = self._one_hot(self.y_int, num_classes=3)

        self.X = self.X[mask]
        self.y = self.y_int[mask]
        self.beta = np.zeros(47, dtype=float)

        self.X_splits = np.array_split(self.X, self.cross_validation)
        self.y_splits = np.array_split(self.y, self.cross_validation)
        

    def save_to_csv(self):

        self.X_no_bias = []
        self.mean = []
        self.std = []
        self.X_norm = []


        for i in range(self.cross_validation):
            self.X_no_bias.append(self.X_splits[i][:, 1:])  # 46 feature
            mean_i = self.X_no_bias[i].mean(axis=0)
            std_i = self.X_no_bias[i].std(axis=0, ddof=0)
            std_i[std_i == 0] = 1.0

            self.mean.append(mean_i)
            self.std.append(std_i)

            X_norm_i = np.concatenate([
                np.ones((self.X_splits[i].shape[0], 1)),
                (self.X_no_bias[i] - mean_i) / std_i
            ], axis=1)

            self.X_norm.append(X_norm_i)


            self.Y_zero_vs_rest.append(np.where(self.y_splits[i] == 0, 1, 0))
            self.Y_one_vs_rest.append(np.where(self.y_splits[i] == 1, 1, 0))
            self.Y_two_vs_rest.append(np.where(self.y_splits[i] == 2, 1, 0))

            x_columns = ['x_0'] + self.home_feature_names + self.away_feature_names
            self.X_df = pd.DataFrame(self.X, columns=x_columns)
            self.X_norm_df = pd.DataFrame(X_norm_i, columns=x_columns)
            self.X_seperate_df = pd.DataFrame(self.X_splits[i], columns=x_columns)
            self.y_df = pd.DataFrame({'y': self.y})
            self.y_seperate_df = pd.DataFrame({'y': self.y_splits[i]})

            self.X_df.to_csv(f'X_matrix.csv', index=False)
            self.y_df.to_csv(f'Y_matrix.csv', index=False)
            self.X_seperate_df.to_csv(f'X_matrix_fold{i}.csv', index=False)
            self.X_norm_df.to_csv(f'X_norm_fold{i}.csv', index=False)
            self.y_seperate_df.to_csv(f'y_fold{i}.csv', index=False)


    
        
        
        


    def build_prediction_vector(self, home_team, away_team, season_idx=None):
        home_team = self._resolve_team(home_team)
        away_team = self._resolve_team(away_team)
        if season_idx is None:
            season_idx = len(self.data_array) - 1

        home_stats = self._find_entry(self.teams_home_statistics_pure, home_team)
        away_stats = self._find_entry(self.teams_away_statistics_pure, away_team)

        home_matches_this_season = int(((self.all_matches_df['season_idx'] == season_idx) & (self.all_matches_df['HomeTeam'] == home_team)).sum())
        away_matches_this_season = int(((self.all_matches_df['season_idx'] == season_idx) & (self.all_matches_df['AwayTeam'] == away_team)).sum())

        home_idx = season_idx * 19 + home_matches_this_season
        away_idx = season_idx * 19 + away_matches_this_season

        a_home_features = self._feature_row_from_side_stats(home_stats, 'home', home_idx).reshape(1, -1)
        b_away_features = self._feature_row_from_side_stats(away_stats, 'away', away_idx).reshape(1, -1)
        final_x_vector = np.concatenate([a_home_features, b_away_features], axis=1)
        return a_home_features, b_away_features, final_x_vector

    

    def run_all(self):
        
        self.load_data()
        self.calculate_scores_and_rates()
        self.process_match_statistics()
        self.handle_missing_seasons()

        self.arrange_data()
        self.build_final_x_matrix()
        self.save_to_csv()


