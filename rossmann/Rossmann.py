import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime
class Rossmann(object):
    def __init__( self ):
        self.home_path=''
        self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb') )
        self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb') )
        self.promo_time_week_scaler        = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb') )
        self.year_scaler                   = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl', 'rb') )
        self.store_type_scaler             = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb') )

    def data_cleaning(self, df):

        # Renomeando as colunas
        cols_old = df.columns
        
        snakecase = lambda x: inflection.underscore( x )
        
        cols_new = list(map(snakecase, cols_old))
        
        # Renomeando
        df.columns = cols_new # !O ERRO ESTÁ AQUI!!! Neste passo ele muda os tipos de dados
        
        # Convertendo os tipos de dados
        df['date'] = pd.to_datetime(df['date'])

        # Imputando uma distância muito grande para o competition distance
        df['competition_distance'] = df['competition_distance'].apply(
            lambda x: 200000.0 if math.isnan(x) else x)

        # Imputando o mês da venda no competition_open_since_month com NA
        df['competition_open_since_month'] = df.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        # Imputando o ano da venda no competition_open_since_year com NA 
        df['competition_open_since_year'] = df.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

        # Imputando a semana do ano da venda para a coluna promo2_since_week com NAs 
        df['promo2_since_week'] = df.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)

        # Imputando a semana do ano da venda para a coluna promo2_since_year com NAs
        df['promo2_since_year'] = df.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        # Criar uma nova coluna (is_promo) para informar, comparando a data da venda (date) com a os meses de intervalo da promoção (promo_interval), se a venda foi feita em período de promo (is_promo)
        month_map = {1: 'Jan',  2: 'Feb',  3: 'Mar',  4: 'Apr',  5: 'May',  6: 'Jun',
                     7: 'Jul',  8: 'Aug',  9: 'Sept',  10: 'Oct', 11: 'Nov', 12: 'Dec'}

        # Imputando valores no intervalo de promoção
        df['promo_interval'].fillna(0, inplace=True)

        # Combinando as datas dos meses com seus números
        df['month_map'] = df['date'].dt.month.map(month_map)

        # Criando a coluna binária de status da promoção
        df['is_promo'] = df.apply(lambda x: 0 if x['promo_interval'] ==
                                  0 else 1 if x['month_map'] in x['promo_interval'] else 0, axis=1)

        # Corrigindo os tipos de dados que ainda falatavam ser corrigidos
        df['competition_open_since_month'] = df['competition_open_since_month'].astype(
            'int64')
        df['competition_open_since_year'] = df['competition_open_since_year'].astype(
            'int64')
        df['promo2_since_week'] = df['promo2_since_week'].astype('int64')
        df['promo2_since_year'] = df['promo2_since_year'].astype('int64')

        return df

    def feature_engineering(self, df03):
        # Criando a variável year
        df03['year'] = df03['date'].dt.year

        # Criando a variável month
        df03['month'] = df03['date'].dt.month

        # Criando a variável day
        df03['day'] = df03['date'].dt.day

        # Criando a variável week_of_year
        df03['week_of_year'] = df03['date'].dt.isocalendar().week

        # Criando a variável year_week
        df03['year_week'] = df03['date'].dt.strftime('%Y-%U')

        # Criando a variável competition_since
        df03['competition_since'] = df03.apply(lambda x: datetime.datetime(
            year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)

        # Criando a variável competition_time_month
        df03['competition_time_month'] = (
            (df03['date'] - df03['competition_since'])/30).apply(lambda x: x.days).astype(int)

        # Criando a variável promo_since
        df03['promo_since'] = df03['promo2_since_year'].astype(
            str) + '-' + df03['promo2_since_week'].astype(str)
        df03['promo_since'] = df03['promo_since'].apply(lambda x: datetime.datetime.strptime(
            x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))

        # Criando a variável promo_time_week
        df03['promo_time_week'] = (
            (df03['date'] - df03['promo_since'])/7).apply(lambda x: x.days).astype(int)

        # Ajustando a variável assortment
        df03['assortment'] = df03['assortment'].apply(
            lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

        # Ajustando a variável state_holiday
        df03['state_holiday'] = df03['state_holiday'].apply(
            lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # Filtrar os dados para lojas abertas e acom vendas maiores que zero
        df03 = df03[df03['open'] != 0]

        # Seleção de colunas
        cols_drop = ['open', 'promo_interval', 'month_map']
        df03 = df03.drop(cols_drop, axis=1)

        return df03

    def data_preparation(self, df06):
        # competition distance
        df06['competition_distance'] = self.competition_distance_scaler.fit_transform(
            df06[['competition_distance']].values)

        # competition time month
        df06['competition_time_month'] = self.competition_time_month_scaler.fit_transform(
            df06[['competition_time_month']].values)

        # promo time week
        df06['promo_time_week'] = self.promo_time_week_scaler.fit_transform(
            df06[['promo_time_week']].values)

        # year
        df06['year'] = self.year_scaler.fit_transform(df06[['year']].values)

        # state_holiday - One Hot Encoding
        df06 = pd.get_dummies(
            df06, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type - Label Encoding
        df06['store_type'] = self.store_type_scaler.fit_transform(
            df06['store_type'])

        # assortment - Ordinal Encoding
        assortment_dict = {'basic': 1,  'extra': 2, 'extended': 3}
        df06['assortment'] = df06['assortment'].map(assortment_dict)

        # day of week
        df06['day_of_week_sin'] = df06['day_of_week'].apply(
            lambda x: np.sin(x * (2. * np.pi/7)))
        df06['day_of_week_cos'] = df06['day_of_week'].apply(
            lambda x: np.cos(x * (2. * np.pi/7)))

        # month
        df06['month_sin'] = df06['month'].apply(
            lambda x: np.sin(x * (2. * np.pi/12)))
        df06['month_cos'] = df06['month'].apply(
            lambda x: np.cos(x * (2. * np.pi/12)))

        # day
        df06['day_sin'] = df06['day'].apply(
            lambda x: np.sin(x * (2. * np.pi/30)))
        df06['day_cos'] = df06['day'].apply(
            lambda x: np.cos(x * (2. * np.pi/30)))

        # week of year
        df06['week_of_year_sin'] = df06['week_of_year'].apply(
            lambda x: np.sin(x * (2. * np.pi/52)))
        df06['week_of_year_cos'] = df06['week_of_year'].apply(
            lambda x: np.cos(x * (2. * np.pi/52)))

        # Seleção de features
        cols_selected_boruta = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos']

        return df06[cols_selected_boruta]

    def get_prediction(self, model, original_data, test_data):
        
        # Predictions
        pred = model.predict(test_data)

        # Join predicted data and the original dataframe
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')
