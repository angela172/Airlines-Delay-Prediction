import streamlit as st
import pandas as pd
from PIL import Image  
import pandas as pd
from plotly_calplot import calplot
import time
from sklearn.metrics import r2_score
import plotly.express as px
import joblib
import base64
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm

init_streamlit_comm()

def sidebar_bg(side_bg):

   side_bg_ext = 'jpg'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
   
side_bg = '/Users/bristi/Desktop/DM assign/Streamlit_app/assets/Background_image2.jpeg'
sidebar_bg(side_bg)

#reading csv file
@st.cache_data
def fetch_flight_data(fltnumber):
    with st.spinner('Loading Data...'):
        time.sleep(0.5)
        if fltnumber == 1:
            flight_data = pd.read_csv("/Users/bristi/Desktop/DM assign/Preprocessing/southwest_test.csv")
        else:
            flight_data = pd.read_csv("/Users/bristi/Desktop/DM assign/Preprocessing/delta_test.csv")
    return flight_data

@st.cache_resource
def get_pyg_renderer() -> "StreamlitRenderer":
    df = pd.read_csv("/Users/bristi/Desktop/DM assign/Preprocessing/Combined_Flights_2022.csv")
    return StreamlitRenderer(df, spec="./gw_config.json", debug=False)

img = Image.open("/Users/bristi/Desktop/DM assign/Streamlit_app/assets/airline.png")
img2 = Image.open("/Users/bristi/Desktop/DM assign/Streamlit_app/assets/airline2.png")


st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"]{
        min-width: 250px;
        max-width: 500px;
    }
    """,
    unsafe_allow_html=True,
)   

st.sidebar.title("Delay Prediction + Flight Analysis")
st.sidebar.image("https://media.giphy.com/media/KenFWffodaDVaceDa6/giphy.gif",width = 100)
st.sidebar.markdown("")
side_bar = st.sidebar.radio('What would you like to view?', [ 'About the dataset', 'Analysis 🔎', 'Predicting Delay','About the model'])

if side_bar == 'About the dataset':
    header = st.container()
    features = st.container()

    with header:
        
        text_col,image_col = st.columns((5.5,1))
        
        with image_col:
            st.write("")
            st.image(img, width = 150)
        with text_col:
            st.title("Understanding the Dataset")
            st.markdown("The dataset contains information of American Airlines from 2018 - 2022.")
            st.write("")
            
            markdown_text = '''
            ```
            Data columns (total 61 columns):
            No   Column                                   Dtype  
            ---  ------                                   -----  
            0   FlightDate                               object 
            1   Airline                                  object 
            2   Origin                                   object 
            3   Dest                                     object 
            4   Cancelled                                bool   
            5   Diverted                                 bool   
            6   CRSDepTime                               int64  
            7   DepTime                                  float64
            8   DepDelayMinutes                          float64
            9   DepDelay                                 float64
            10  ArrTime                                  float64
            11  ArrDelayMinutes                          float64
            12  AirTime                                  float64
            13  CRSElapsedTime                           float64
            14  ActualElapsedTime                        float64
            15  Distance                                 float64
            16  Year                                     int64  
            17  Quarter                                  int64  
            18  Month                                    int64  
            19  DayofMonth                               int64  
            20  DayOfWeek                                int64  
            21  Marketing_Airline_Network                object 
            22  Operated_or_Branded_Code_Share_Partners  object 
            23  DOT_ID_Marketing_Airline                 int64  
            24  IATA_Code_Marketing_Airline              object 
            25  Flight_Number_Marketing_Airline          int64  
            26  Operating_Airline                        object 
            27  DOT_ID_Operating_Airline                 int64  
            28  IATA_Code_Operating_Airline              object 
            29  Tail_Number                              object 
            30  Flight_Number_Operating_Airline          int64  
            31  OriginAirportID                          int64  
            32  OriginAirportSeqID                       int64  
            33  OriginCityMarketID                       int64  
            34  OriginCityName                           object 
            35  OriginState                              object 
            36  OriginStateFips                          int64  
            37  OriginStateName                          object 
            38  OriginWac                                int64  
            39  DestAirportID                            int64  
            40  DestAirportSeqID                         int64  
            41  DestCityMarketID                         int64  
            42  DestCityName                             object 
            43  DestState                                object 
            44  DestStateFips                            int64  
            45  DestStateName                            object 
            46  DestWac                                  int64  
            47  DepDel15                                 float64
            48  DepartureDelayGroups                     float64
            49  DepTimeBlk                               object 
            50  TaxiOut                                  float64
            51  WheelsOff                                float64
            52  WheelsOn                                 float64
            53  TaxiIn                                   float64
            54  CRSArrTime                               int64  
            55  ArrDelay                                 float64
            56  ArrDel15                                 float64
            57  ArrivalDelayGroups                       float64
            58  ArrTimeBlk                               object 
            59  DistanceGroup                            int64  
            60  DivAirportLandings                       int64 
            ```
            '''
            st.markdown(markdown_text)
            st.write("")    

elif side_bar == 'Analysis 🔎':
    st.title("Analysing Flight Data 2022")
    st.write("Using PyGWalker to perform EDA on the combined flights dataset")
    st.write("")
    
    renderer = get_pyg_renderer()
    renderer.render_explore()

elif side_bar == "Predicting Delay":  
     
    text,img2 = st.columns((2,1))
    with text:
        st.title("Predicting by how many minutes a flight will be delayed")
        st.markdown("Trained on historical data from `2018-2021`") 
        st.markdown("Predicts `2022` flight") 
        st.write("")
        st.write("")
        
    with img2:
        st.image("https://media.giphy.com/media/fsDIB8vYY4Vq2OSVwt/giphy.gif", width = 150)
        
    st.write("We have identified that SouthWestern Airlines and Delta Airlines are the most \
            popular and have maximum data available in the dataset.")
    
    st.write("In this page, you can choose between these two airlines and \
        use our model to predict by hom many minutes flights will get delayed all through out the year.")
    
    st.write("⚠️ Please note that these airlines have multiple flights going out on the same day.\
        So, predicted delay and actual delay are averaged.")
    
    options = ["Southwestern Airlines", "Delta Airlines"]
    options2 = ["Linear Regression", "Random Forest", "Decision Tree", "XGBoost" ]
    with st.form(key='airline_form'):
        selected_option = st.selectbox("Select an Airline", options)
        selected_option2 = st.selectbox("Select a Model: ", options2)
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        markdown_text = f" Airline selected: `{selected_option}\n`"
        st.sidebar.markdown(markdown_text)
        markdown_text = f" Model selected: `{selected_option2}\n`"
        st.sidebar.markdown(markdown_text)
        
        if selected_option == "Delta Airlines":
            if selected_option2 == "Linear Regression":
                loaded_model = joblib.load('/Users/bristi/Desktop/DM assign/Streamlit_app/assets/delta_LR.pkl')

            if selected_option2 == "Random Forest":
                loaded_model = joblib.load('/Users/bristi/Desktop/DM assign/Preprocessing/rf_delta.pkl')
            
            if selected_option2 == "Decision Tree":
                loaded_model = joblib.load('/Users/bristi/Desktop/DM assign/Streamlit_app/assets/dt_delta.pkl')
             
            if selected_option2 == "XGBoost":
                loaded_model = joblib.load('/Users/bristi/Desktop/DM assign/Streamlit_app/assets/xgb_delta.pkl')
                
            df_test = fetch_flight_data(0)
            df_notcancelled = df_test[df_test["Cancelled"] == False]
            df_notcancelled = df_notcancelled.drop('Cancelled', axis = 1)
            df_test = df_notcancelled[df_notcancelled["Diverted"] == False]
            null_dep_time_count = df_test["DepTime"].isnull().sum()
            enc_nom_1 = (df_test.groupby('Origin').size()) / len(df_test)
            df_test['Origin_enc'] = df_test['Origin'].apply(lambda x : enc_nom_1[x])
            enc_nom_2 = (df_test.groupby('Dest').size()) / len(df_test)
            df_test['Dest_enc'] = df_test['Dest'].apply(lambda x : enc_nom_2[x])
            X_test = df_test[['DepDelayMinutes', 'TaxiOut', 'TaxiIn', 'Distance', 'ArrTime', 'Origin_enc', 'CRSArrTime', 'DayofMonth', 'DepTime']]
            y_test = df_test['ArrDelayMinutes']
            
        if selected_option == "Southwestern Airlines":
            if selected_option2 == "Linear Regression":
                loaded_model = joblib.load('/Users/bristi/Desktop/DM assign/Streamlit_app/assets/linearReg_southwest.pkl')

            if selected_option2 == "Random Forest":
                loaded_model = joblib.load('/Users/bristi/Desktop/DM assign/Streamlit_app/assets/rf_sw.pkl')
            
            if selected_option2 == "Decision Tree":
                loaded_model = joblib.load('/Users/bristi/Desktop/DM assign/Streamlit_app/assets/dt_sw.pkl')
             
            if selected_option2 == "XGBoost":
                loaded_model = joblib.load('/Users/bristi/Desktop/DM assign/Streamlit_app/assets/xgb_sw.pkl')
            
            df_test = fetch_flight_data(1)
            df_notcancelled = df_test[df_test["Cancelled"] == False]
            df_test = df_notcancelled[df_notcancelled["Diverted"] == False]
            X_test = df_test[['DepDelayMinutes', 'TaxiOut', 'TaxiIn', 'ArrTime', 'Distance']]
            y_test = df_test['ArrDelayMinutes']
                    
        pred = loaded_model.predict(X_test)
        r2 = r2_score(y_test, pred)
        markdown_text = f" R2 Score: `{r2}\n`"
        st.write("")
        st.markdown(markdown_text)
        st.write("")
        
        # visualisation
        pred = pd.DataFrame(pred)
        actual = pd.concat([ df_test[['FlightDate']].reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        predicted = pd.concat([ df_test[['FlightDate']].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)

        df5 = predicted
        average_values = df5.groupby('FlightDate')[0].mean().reset_index()
        average_values.rename(columns={0: 'Average_ArrivalDelay'}, inplace=True)
        merged_df = pd.merge(df5, average_values, on='FlightDate', how='left')
        final_predicted = merged_df.drop_duplicates(subset='FlightDate')
        final_predicted = final_predicted.drop([0], axis=1)
        final_predicted = final_predicted.sort_values(by='FlightDate', ascending=True)
        
        # Average actual delay of flights on a single day
        df6 = actual
        average_values = df6.groupby('FlightDate')['ArrDelayMinutes'].mean().reset_index()
        average_values.rename(columns={'ArrDelayMinutes': 'Average_ArrivalDelay'}, inplace=True)
        merged_df = pd.merge(df6, average_values, on='FlightDate', how='left')
        final_actual = merged_df.drop_duplicates(subset='FlightDate')
        final_actual = final_actual.drop(['ArrDelayMinutes'], axis=1)
        final_actual = final_actual.sort_values(by='FlightDate', ascending=True)

        start_date = '2022-08-01'
        end_date = '2022-12-31'

        missing_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        missing_data = pd.DataFrame({
            'FlightDate': missing_dates,
            'Average_ArrivalDelay': -1
        })

        final_actual['FlightDate'] = pd.to_datetime(final_actual['FlightDate'])
        missing_data['FlightDate'] = pd.to_datetime(missing_data['FlightDate'])
        final_actual = pd.concat([final_actual, missing_data])
        final_actual = final_actual.sort_values('FlightDate').reset_index(drop=True)
        
        fig1 = calplot(
            final_actual, 
            x= "FlightDate", 
            y = "Average_ArrivalDelay",
            dark_theme=False,
            years_title=True,
            gap=1,
            name="ArrDelayMinutes",
            month_lines_width=2, 
            month_lines_color="black",
            colorscale="magma",
            showscale=True,
            cmap_max=69,
            cmap_min= 0
            )

        final_actual['FlightDate'] = pd.to_datetime(final_actual['FlightDate'])
        date_formatted = final_actual['FlightDate'].dt.strftime('%Y-%m-%d')

        fig1.update_layout(title_text=f'<b>Actual Average ArrivalDelay for flights</b>')
        fig1.update_layout(width=800)


        st.plotly_chart(fig1)
        st.write("")
        
        final_predicted['FlightDate'] = pd.to_datetime(final_predicted['FlightDate'])
        missing_data['FlightDate'] = pd.to_datetime(missing_data['FlightDate'])

        final_predicted = pd.concat([final_predicted, missing_data])

        final_predicted = final_predicted.sort_values('FlightDate').reset_index(drop=True)
        
        fig2 = calplot(
            final_predicted, 
            x= "FlightDate", 
            y = "Average_ArrivalDelay",
            dark_theme=False,
            years_title=True,
            gap=1,
            name="ArrDelayMinutes",
            month_lines_width=2, 
            month_lines_color="black",
            colorscale="magma",
            showscale=True,
            cmap_max=69,
            cmap_min= 0
            )

        final_predicted['FlightDate'] = pd.to_datetime(final_predicted['FlightDate'])
        date_formatted = final_predicted['FlightDate'].dt.strftime('%Y-%m-%d')

        fig2.update_layout(title_text=f'<b>Predicted Average ArrivalDelay for flights</b>')
        fig2.update_layout(width=800)

        st.plotly_chart(fig2)
        st.write("")
        
        final_diff = final_predicted
        final_diff['Average_ArrivalDelay'] = final_predicted['Average_ArrivalDelay'] - final_actual['Average_ArrivalDelay']

        fig3 = calplot(
            final_diff, 
            x= "FlightDate", 
            y = "Average_ArrivalDelay",
            dark_theme=False,
            years_title=True,
            gap=1,
            name="ArrDelayMinutes",
            month_lines_width=2, 
            month_lines_color="black",
            colorscale="magma",
            showscale=True
            # cmap_max= 2,
            # cmap_min= -60
            )

        final_diff['FlightDate'] = pd.to_datetime(final_diff['FlightDate'])
        date_formatted = final_diff['FlightDate'].dt.strftime('%Y-%m-%d')

        fig3.update_layout(title_text=f'<b>Delta Average ArrivalDelay for flights</b>')
        fig3.update_layout(width=800)

        st.plotly_chart(fig3)
        st.write("")
        

        fig4 = px.scatter(final_diff, x='FlightDate', y='Average_ArrivalDelay', title='Arrival Delay vs Flight Date')
        fig4.update_layout(xaxis_title='Flight Date', yaxis_title='Average Arrival Delay')

        st.plotly_chart(fig4)
        st.write("")

elif side_bar == "About the model":
    st.title("About the model")
    markdown2='''
    The model built in this project helps us predict the delay in the arrival of two airlines mainly - Southwest Airlines and Delta airlines.
    The dataset for the model has been derived from Kaggle. The model has been trained on the 2018-2021 datasets for both the airlines and for testing the 
    2022 dataset has been used for both the airlines. 
    
    The features for building the model has been derived from the respective original datasets using `Random Forest feature selection.` 
    
    Using this technique the different features obtained for the Delta Airlines are:
    - DepDelayMinutes: The delay in departure time in minutes (float)
    - TaxiOut: Time taken for taxiing out from the departure airport.
    - TaxiIn: Time taken for taxiing in after arrival.
    - Distance: To analyze if longer flights tend to have more delays.
    - ArrTime: The actual arrival time (in float format).
    - Origin_enc: The code or name of the origin airport(encoded).
    - CRSArrTime: The scheduled arrival time.
    - DayofMonth: The day of the month when the flight occurred.
    - DepTime: The actual departure time (in float format).
    
    Similarly the set of features obtained for Southwest Airlines:
    - DepDelayMinutes: The delay in departure time in minutes (float).
    - TaxiOut: Time taken for taxiing out from the departure airport.
    - TaxiIn: Time taken for taxiing in after arrival.
    - ArrTime: The actual arrival time (in float format).
    - Distance: The distance traveled by the flight (in float format).
    
    These features are then used to train the model so as to predict `ArrDelayMinutes` for both airlines. 
    
    Moreover, these features have also been used to build a `correlation matrix` which shows the dependancies or the correlations 
    of different features. It has been ensured that 
    
    '''
    st.markdown(markdown2)
    img6 = Image.open("/Users/bristi/Desktop/DM assign/Streamlit_app/assets/newplot.png")
    st.image(img6, caption=" ", use_column_width=True)
    st.write("")
    
    markdown3= '''
    Moving on, these features have been used to build four models for the both the airlines namely, `Linear Regression`, `XGBoost`, `Decision Tree Regression` and `Random Forest Regression`.
    Out of these four models, `Random Forest Regression` has proven to be a better performing model for both the airlines after Hyperparameter tuning giving an r2 value of `0.967` for Southwest Airlines and `0.985` for Delta Airlines.
'''
    st.markdown(markdown3)
    st.write("")