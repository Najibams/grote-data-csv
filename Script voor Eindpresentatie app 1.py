import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static
from geopy.distance import great_circle
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from folium import CircleMarker
from sklearn.tree import DecisionTreeRegressor


# Functies om data te laden met caching
@st.cache_data
def load_weather_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_journey_data(file_path):
    return pd.read_csv(file_path)

# Bestanden inlezen
weather_file = r"C:\Users\Jazz Fust\Downloads\weather_new.csv"
journey_file = r"C:\Users\Jazz Fust\Downloads\147JourneyDataExtract30Jan2019-05Feb2019.csv"
stations_file = r"C:\Users\Jazz Fust\Downloads\cycle_stations.csv"
london_stations_file = r"C:\Users\Jazz Fust\Downloads\London stations.csv"

# Weather data inlezen
weather_df = load_weather_data(weather_file)
weather_df['Unnamed: 0'] = pd.to_datetime(weather_df['Unnamed: 0'])
weather_df = weather_df[(weather_df['Unnamed: 0'] >= '2019-01-30') & (weather_df['Unnamed: 0'] <= '2019-02-05')]
weather_df = weather_df[['Unnamed: 0', 'tavg']]
weather_df.columns = ['date', 'avg_temp']

# Journey data inlezen
journey_df = load_journey_data(journey_file)
journey_df['Start Date'] = pd.to_datetime(journey_df['Start Date'], format='%d/%m/%Y %H:%M')
journey_df['End Date'] = pd.to_datetime(journey_df['End Date'], format='%d/%m/%Y %H:%M')

# Stations data inlezen
stations_df = pd.read_csv(stations_file)

# London stations data inlezen
london_stations_df = pd.read_csv(london_stations_file)
london_stations_df.columns = london_stations_df.columns.str.strip()  # Verwijder spaties voor en achter

# Streamlit app
st.title("🚴‍♂️ Fietsreizen en Weerdata 🌤️")
st.markdown("Een interactief dashboard voor het analyseren van fietsreizen en weerdata.")

# Dropdown menu voor dashboard selectie
dashboard_option = st.selectbox("Selecteer een optie:", ["Introductie", "Oud Dashboard", "Nieuw Dashboard", "Conclusie"])

if dashboard_option == "Introductie":
    st.subheader("Introductie")
    st.write("""
        In dit project hebben we ons gericht op de volgende punten:
        - 📊 We hebben **case 3** aangepast, omdat we geloven dat hier de grootste verbetering mogelijk is.
        - 🎨 We hebben de **visualisaties** en de algemene presentatie van het dashboard verbeterd om de gebruiksvriendelijkheid te vergroten.
        - 📈 De **forecast** is geoptimaliseerd met verschillende modellen.
        - 🗺️ De **kaart** is zo interactief mogelijk gemaakt met diverse aantrekkelijke opties.
        - 🔍 Voorafgaand hebben we een andere vorm van **data-exploratie** uitgevoerd, zodat we beter inzicht hadden in welke data we moesten gebruiken.
    """)


elif dashboard_option == "Oud Dashboard":
    # Sidebar filters voor het oude dashboard
    st.sidebar.header("🔍 Data Analyse")
    show_boxplots = st.sidebar.checkbox("Toon boxplots na opschonen", value=False)

    # Datum selectie
    st.sidebar.subheader("📅 Selecteer Datums")
    start_date = st.sidebar.date_input("Selecteer startdatum", value=pd.to_datetime('2019-01-30'), min_value=pd.to_datetime('2019-01-30'), max_value=pd.to_datetime('2019-02-05'))
    end_date = st.sidebar.date_input("Selecteer einddatum", value=pd.to_datetime('2019-02-05'), min_value=pd.to_datetime('2019-01-30'), max_value=pd.to_datetime('2019-02-05'))

    # Slider voor temperatuur range
    st.sidebar.subheader("🌡️ Selecteer Temperatuur Range (°C)")
    temp_min, temp_max = st.sidebar.slider("Temperatuur", min_value=int(weather_df['avg_temp'].min()), max_value=int(weather_df['avg_temp'].max()), value=(0, 10))

    # Grafiek opties
    st.sidebar.subheader("📈 Selecteer Analyse Type")
    forecast_checkbox_temp = st.sidebar.checkbox("Temperatuurvoorspelling toevoegen", value=False)
    forecast_checkbox_rides = st.sidebar.checkbox("Voorspelling toevoegen aan Bike Journey Analysis", value=True)

    # Filter journey data op datum
    journey_df_filtered = journey_df[(journey_df['Start Date'].dt.date >= start_date) & (journey_df['Start Date'].dt.date <= end_date)]

    # Merge met stations data
    journey_with_start = journey_df_filtered.merge(stations_df, left_on='StartStation Name', right_on='name', how='left')
    journey_with_stations = journey_with_start.merge(stations_df, left_on='EndStation Name', right_on='name', how='left', suffixes=('', '_end'))
    journey_with_stations.rename(columns={'lat': 'lat_start', 'long': 'long_start'}, inplace=True)
    journey_with_stations.dropna(subset=['lat_start', 'long_start', 'lat_end', 'long_end'], inplace=True)

    # Merge met weather data
    merged_data = journey_with_stations.merge(weather_df, left_on='Start Date', right_on='date', how='left')
    merged_data.dropna(subset=['avg_temp'], inplace=True)

    # Bereken de afstand
    merged_data['distance'] = merged_data.apply(lambda row: great_circle((row['lat_start'], row['long_start']), (row['lat_end'], row['long_end'])).meters, axis=1)

    # Filter data op temperatuur
    filtered_for_graphs = merged_data[(merged_data['avg_temp'] >= temp_min) & (merged_data['avg_temp'] <= temp_max)]

    # Pagina logica voor het oude dashboard
    st.subheader("📈 Analyse")
    
    # Boxplots of overige analyses
    if show_boxplots:
        st.subheader("📊 Boxplot van Temperatuur en Aantal Fietstochten")
        fig_temp_box = px.box(weather_df, y='avg_temp', title='Boxplot van Gemiddelde Temperatuur (°C)', points=False)
        ride_counts = journey_df.groupby(journey_df['Start Date'].dt.date).size().reset_index(name='Number of Rides')
        fig_rides_box = px.box(ride_counts, y='Number of Rides', title='Boxplot van Aantal Fietstochten', points=False)
        fig_box = make_subplots(rows=1, cols=2, subplot_titles=('Gemiddelde Temperatuur', 'Aantal Fietstochten'))

        for trace in fig_temp_box.data:
            fig_box.add_trace(trace, row=1, col=1)

        for trace in fig_rides_box.data:
            fig_box.add_trace(trace, row=1, col=2)

        fig_box.update_layout(title_text='Boxplots van Temperatuur en Aantal Fietstochten', showlegend=False)
        st.plotly_chart(fig_box)
        st.write("De data is goed opgeschoond, waardoor outliers niet meer zichtbaar zijn.")

    # Grafiek opties voor analyses
    else:
        plot_type = st.sidebar.selectbox("Selecteer type analyse", ["Gemiddelde Temperatuur", "Bike Journey Analysis"])
    
        if plot_type == "Gemiddelde Temperatuur":
            st.subheader("🌡️ Gemiddelde Temperatuur over de Geselecteerde Data")
            avg_temp_by_date = filtered_for_graphs.groupby('date')['avg_temp'].mean().reset_index()
            fig_temp = px.line(avg_temp_by_date, x='date', y='avg_temp', title='Gemiddelde Temperatuur (°C)', markers=True)
            fig_temp.update_layout(xaxis_title="Datum", yaxis_title="Gemiddelde Temperatuur (°C)",
                                   yaxis=dict(range=[avg_temp_by_date['avg_temp'].min() - 5, avg_temp_by_date['avg_temp'].max() + 5]),
                                   showlegend=True)

            if forecast_checkbox_temp:
                # Lineaire regressie voor temperatuurvoorspelling
                X_temp = np.array((avg_temp_by_date['date'] - avg_temp_by_date['date'].min()).dt.days).reshape(-1, 1)
                y_temp = avg_temp_by_date['avg_temp'].values
                model_temp = LinearRegression()
                model_temp.fit(X_temp, y_temp)

                # Voorspelling voor 6, 7, 8 en 9 februari
                future_dates = pd.date_range(start='2019-02-06', end='2019-02-09')
                future_X_temp = np.array((future_dates - avg_temp_by_date['date'].min()).days).reshape(-1, 1)
                predicted_temps = model_temp.predict(future_X_temp)
                future_data = pd.DataFrame({'date': future_dates, 'avg_temp': predicted_temps})

                # Combineer historische en voorspelde data
                combined_data = pd.concat([avg_temp_by_date, future_data]).reset_index(drop=True)
                fig_temp = px.line(combined_data, x='date', y='avg_temp', title='Gemiddelde Temperatuur (°C)', markers=True)
                fig_temp.add_scatter(x=future_data['date'], y=future_data['avg_temp'], mode='markers+lines', name='Voorspelde Temperatuur', line=dict(color='red'))

            st.plotly_chart(fig_temp)

        elif plot_type == "Bike Journey Analysis":
            st.header("🚴‍♂️ Bike Journey Analysis")
            journey_df_filtered['Total duration (min)'] = journey_df_filtered['Duration'] / 60
            journey_df_filtered['Day'] = journey_df_filtered['Start Date'].dt.date
            daily_rides = journey_df_filtered.groupby('Day').size().reset_index(name='Number of Rides')
            daily_rides['Day'] = pd.to_datetime(daily_rides['Day'])
            fig = px.line(daily_rides, x='Day', y='Number of Rides', title='📈 Daily Bike Rides', markers=True)

            # Voorspelling voor het aantal ritten
            if forecast_checkbox_rides:
                X_rides = (daily_rides['Day'] - daily_rides['Day'].min()).dt.days.values.reshape(-1, 1)
                y_rides = daily_rides['Number of Rides'].values
                model_rides = LinearRegression()
                model_rides.fit(X_rides, y_rides)
                future_days_rides = np.array([(daily_rides['Day'].max() + pd.Timedelta(days=i) - daily_rides['Day'].min()).days for i in range(1, 6)]).reshape(-1, 1)
                predicted_rides = model_rides.predict(future_days_rides)
                future_dates_rides = [daily_rides['Day'].max() + pd.Timedelta(days=i) for i in range(1, 6)]
                future_rides_data = pd.DataFrame({'Day': future_dates_rides, 'Number of Rides': predicted_rides})

                # Combineer historische en voorspelde data
                fig.add_scatter(x=future_rides_data['Day'], y=future_rides_data['Number of Rides'], mode='markers+lines',
                                 name='Voorspelde Ritten', line=dict(color='orange', width=2))  # Kleur gewijzigd naar oranje

            fig.update_layout(showlegend=True)
            st.plotly_chart(fig)

elif dashboard_option == "Nieuw Dashboard":
    
    # Bestanden inlezen
    file1 = r"C:\Users\Jazz Fust\Downloads\96JourneyDataExtract07Feb2018-13Feb2018.csv"
    file2 = r"C:\Users\Jazz Fust\Downloads\97JourneyDataExtract14Feb2018-20Feb2018.csv"
    file3 = r"C:\Users\Jazz Fust\Downloads\98JourneyDataExtract21Feb2018-27Feb2018.csv"
    file4 = r"C:\Users\Jazz Fust\Downloads\99JourneyDataExtract28Feb2018-06Mar2018.csv"
    
    # DataFrames inlezen
    file1_df = pd.read_csv(file1)
    file2_df = pd.read_csv(file2)
    file3_df = pd.read_csv(file3)
    file4_df = pd.read_csv(file4)
    
    # DataFrames samenvoegen
    combined_df = pd.concat([file1_df, file2_df, file3_df, file4_df], ignore_index=True)
    
    # Opschonen van de gecombineerde DataFrame
    cleaned_data_journey = combined_df[combined_df['Duration'] <= 1900]
    
    # Inlezen van de stationsgegevens
    stations_file = r"C:\Users\Jazz Fust\Downloads\cycle_stations.csv"
    london_stations_file = r"C:\Users\Jazz Fust\Downloads\London stations.csv"
    
    # Stationsgegevens inlezen
    stations_df = pd.read_csv(stations_file)
    london_stations_df = pd.read_csv(london_stations_file)
    
    # Merge voor StartStation Id
    merged_start = pd.merge(cleaned_data_journey, stations_df, left_on='StartStation Id', right_on='id', suffixes=('', '_start'))
    
    # Hernoem de kolommen die betrekking hebben op het startstation
    merged_start = merged_start.rename(columns={
        'name': 'name_start',
        'terminalName': 'terminalName_start',
        'lat': 'lat_start',
        'long': 'long_start',
        'installed': 'installed_start',
        'locked': 'locked_start',
        'installDate': 'installDate_start',
        'removalDate': 'removalDate_start',
        'temporary': 'temporary_start',
        'nbBikes': 'nbBikes_start'
    })
    
    # Merge voor EndStation Id
    merged_data = pd.merge(merged_start, stations_df, left_on='EndStation Id', right_on='id', suffixes=('', '_end'))
    
    # Hernoem de kolommen die betrekking hebben op het eindstation
    merged_data = merged_data.rename(columns={
        'name': 'name_end',
        'terminalName': 'terminalName_end',
        'lat': 'lat_end',
        'long': 'long_end',
        'installed': 'installed_end',
        'locked': 'locked_end',
        'installDate': 'installDate_end',
        'removalDate': 'removalDate_end',
        'temporary': 'temporary_end',
        'nbBikes': 'nbBikes_end'
    })
    
    # Verwijder dubbele id kolommen na merge
    merged_data = merged_data.drop(columns=['id', 'id_end'])
    
    # Opschonen van de merged_data
    cleaned_data_merged = merged_data.dropna(axis=1)
    
    # Weather data inlezen
    weather_file = r"C:\Users\Jazz Fust\Downloads\weather_new.csv"
    weather_df = pd.read_csv(weather_file)
    
    # Weather data opschonen
    weather_df['Unnamed: 0'] = pd.to_datetime(weather_df['Unnamed: 0'])
    weather_df = weather_df[(weather_df['Unnamed: 0'] >= '2018-02-07') & (weather_df['Unnamed: 0'] <= '2018-03-06')]
    weather_df['Date'] = weather_df['Unnamed: 0']
    weather_df = weather_df.drop(columns=['Unnamed: 0'])
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    cleaned_data_weather = weather_df.dropna(axis=1)
    
    # Converteer de datumkolommen naar datetime-formaat
    cleaned_data_merged['Start Date'] = pd.to_datetime(cleaned_data_merged['Start Date'])
    cleaned_data_merged['End Date'] = pd.to_datetime(cleaned_data_merged['End Date'])
    
    # Gebruik normalize() om alleen de datum te krijgen (zonder tijd)
    cleaned_data_merged['Start_Date_Only'] = cleaned_data_merged['Start Date'].dt.normalize()
    cleaned_data_merged['End_Date_Only'] = cleaned_data_merged['End Date'].dt.normalize()
    
    # Merge op basis van Start Date
    merged_start = pd.merge(cleaned_data_merged, cleaned_data_weather, left_on='Start_Date_Only', right_on='Date', how='left', suffixes=('', '_start'))
    
    # Merge op basis van End Date
    journey_df = pd.merge(merged_start, cleaned_data_weather, left_on='End_Date_Only', right_on='Date', how='left', suffixes=('', '_end'))
    
    # Verwijder de tijdelijke datumkolommen
    journey_df = journey_df.drop(columns=['Start_Date_Only', 'End_Date_Only', 'Date', 'Date_end'])

    journey_df['Start Date'] = pd.to_datetime(journey_df['Start Date'], format="%Y-%m-%d %H:%M:%S")
    journey_df['End Date'] = pd.to_datetime(journey_df['End Date'], format="%Y-%m-%d %H:%M:%S")

    # Groepeer de journey_df op de 'Start Date' en tel het aantal ritten per dag
    daily_rides_df = journey_df.groupby(journey_df['Start Date'].dt.date).size().reset_index(name='Number of Rides')

    # Hernoem de kolommen voor duidelijkheid
    daily_rides_df.columns = ['Date', 'Number of Rides']

    # Zet de 'Date' kolom in daily_rides_df om naar datetime
    daily_rides_df['Date'] = pd.to_datetime(daily_rides_df['Date'])

    # Merge de DataFrames
    merged_df = pd.merge(daily_rides_df, weather_df, on='Date')

    # Groepeer op start_station_name en tel het aantal ritten
    start_station_rides_df = journey_df.groupby(['name_start', 'long_start', 'lat_start']).size().reset_index(name='Number of Rides')
    end_station_rides_df = journey_df.groupby(['name_end', 'long_end', 'lat_end']).size().reset_index(name='Number of Rides')

    # Start van het dashboard
    st.title("🚴‍♂️ Fietsreizen in relatie met weerdata in Londen🌤️")
    st.markdown("Een interactief dashboard voor het analyseren van fietsreizen en weerdata.")
    
    tab_selection = st.tabs(["Welkom", "Analyse", "Voorspellingen", "Kaart","Conclusie"])

    with tab_selection[0]:
        st.subheader("Welkom bij ons Dashboard :wave:")
        
        st.write("""
                 Welkom! In dit dashboard nemen we je mee in de invloed van het weer op het aantal fietsritten in Londen. 
                 We onderzoeken hoe weersomstandigheden, zoals temperatuur en neerslag, van invloed zijn op het gebruik van 
                 openbare fietsen in de stad. 
                 
                 Onze motivatie voor dit onderzoek komt voort uit onze eigen ervaringen met het openbaar vervoer in Londen. 
                 We willen dieper ingaan op de relatie tussen het weer en het fietsgebruik, en aan de hand van deze analyses 
                 doen we voorspellingen over het aantal fietsritten in de toekomst.
                 
                 We hopen dat dit dashboard niet alleen inzicht biedt in het huidige gebruik van fietsen tijdens verschillende 
                 weersomstandigheden, maar ook waardevolle informatie levert voor beleidsmakers en fietsvoorzieningen in de stad.
                 """)
    
    with tab_selection[1]:
        st.subheader("📈 Analyse")
        
        # Verwijder de kolommen 'date' en 'Unnamed: 0.1' uit de DataFrame
        columns_to_remove = ['date', 'Unnamed: 0.1']
        merged_df = merged_df.drop(columns=[col for col in columns_to_remove if col in merged_df.columns])
        
        # Bereken de correlatiematrix
        correlation_matrix = merged_df.corr()
        
        # Selecteer alleen de kolom 'Number of Rides' en de weerdata
        rides_correlation = correlation_matrix[['Number of Rides']]
        
        # Draai de correlatiematrix
        rides_correlation = rides_correlation.T  # Transpose to have rides on one side
        
        # Toon de correlatiematrix als een heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(rides_correlation, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
        plt.title('Correlatiematrix van Aantal Ritten en Weerdata', color='white')
        plt.gca().set_facecolor('black')
        
        # Toont de correlatiematrix in Streamlit
        st.pyplot(plt)
    
        # Checkbox voor regressieplots
        show_regression = st.checkbox("Geef een regressieplot van de grootste correlatiewaarde")
    
        if show_regression:
            # Maak regressieplots voor de kolommen met de hoogste correlatie
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
            
            # Regressieplot voor 'tmin'
            sns.regplot(x='tmin', y='Number of Rides', data=merged_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=axs[0])
            axs[0].set_title('Regressieplot van Minimum Temperatuur en Aantal Ritten')
            axs[0].set_xlabel('Minimum Temperatuur (tmin)')
            axs[0].set_ylabel('Aantal Ritten')
    
            # Regressieplot voor 'tavg' (gemiddelde temperatuur)
            sns.regplot(x='tavg', y='Number of Rides', data=merged_df, scatter_kws={'alpha':0.5}, line_kws={'color':'blue'}, ax=axs[1])
            axs[1].set_title('Regressieplot van Gemiddelde Temperatuur en Aantal Ritten')
            axs[1].set_xlabel('Gemiddelde Temperatuur (tavg)')
            axs[1].set_ylabel('Aantal Ritten')
            
            # Toont de regressieplots in Streamlit
            st.pyplot(fig)

    with tab_selection[2]:
        st.subheader("📈 Voorspellingen")
        
        # Info box over voorspellingen
        st.info("Kies een voorspellingstype en pas het aantal dagen aan. Gebruik de schakelaar om voorspellingen aan/uit te zetten.")
        
        # Schakelaar om voorspellingen aan/uit te zetten
        toggle_predictions = st.checkbox("Toon Voorspellingen", value=True)
        
        # Voorspellingstype kiezen
        prediction_option = st.selectbox(
            "Kies een voorspellingstype", 
            ["Aantal Ritten Voorspellingen", "Gemiddelde Temperatuur Voorspellingen", "Gecombineerde Voorspelling"]
        )
        
        # Aantal dagen voor voorspelling kiezen
        prediction_days = st.slider("Aantal Voorspellingsdagen", min_value=7, max_value=60, value=30, step=7)
        
        # Bepaal de laatste datum in de dataset
        last_date = merged_df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
        
        # Model voor aantal ritten voorspellingen
        X_rides = merged_df[['tavg']]
        y_rides = merged_df['Number of Rides']
        
        model_rides = LinearRegression()
        model_rides.fit(X_rides, y_rides)
        
        future_tavg_rides = np.random.uniform(low=merged_df['tavg'].min(), high=merged_df['tavg'].max(), size=(prediction_days, 1))
        future_rides = model_rides.predict(future_tavg_rides)
        
        # Model voor gemiddelde temperatuur voorspellingen
        X_temp = merged_df[['Number of Rides']]
        y_temp = merged_df['tavg']
        
        model_temp = LinearRegression()
        model_temp.fit(X_temp, y_temp)
        
        future_rides_temp = np.random.uniform(low=merged_df['Number of Rides'].min(), high=merged_df['Number of Rides'].max(), size=(prediction_days, 1))
        future_tavg = model_temp.predict(future_rides_temp)
        
        # Voorspellingen DataFrame
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Number of Rides': future_rides,
            'Predicted Average Temperature (tavg)': future_tavg
        })
        
        # Statistische gegevens
        gemiddelde_tavg = np.mean(predictions_df['Predicted Average Temperature (tavg)'])
        mediaan_tavg = np.median(predictions_df['Predicted Average Temperature (tavg)'])
        std_tavg = np.std(predictions_df['Predicted Average Temperature (tavg)'])
        
        # Toon statistieken in aparte kaarten
        st.metric("Gemiddelde Temperatuur", f"{gemiddelde_tavg:.2f} °C")
        st.metric("Mediaan Temperatuur", f"{mediaan_tavg:.2f} °C")
        st.metric("Standaarddeviatie Temperatuur", f"{std_tavg:.2f} °C")
        
        # Download knop
        if st.button("Download Voorspellingen als CSV"):
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="voorspellingen.csv",
                mime="text/csv",
            )
        
        if prediction_option == "Aantal Ritten Voorspellingen":
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(merged_df['Date'], merged_df['Number of Rides'], label='Aantal Ritten (Historisch)', color='blue')
            if toggle_predictions:
                ax.plot(predictions_df['Date'], predictions_df['Predicted Number of Rides'], label='Aantal Ritten (Voorspelling)', color='orange', linestyle='--')
            ax.axvline(x=last_date, color='gray', linestyle='--', label='Laatste Datum')
            plt.title('Aantal Ritten Voorspellingen')
            plt.xlabel('Datum')
            plt.ylabel('Aantal Ritten')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
            st.pyplot(fig)
        
        elif prediction_option == "Gemiddelde Temperatuur Voorspellingen":
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(merged_df['Date'], merged_df['tavg'], label='Gemiddelde Temperatuur (Historisch)', color='green')
            if toggle_predictions:
                ax.plot(predictions_df['Date'], predictions_df['Predicted Average Temperature (tavg)'], label='Temperatuur (Voorspelling)', color='red', linestyle='--')
            ax.axvline(x=last_date, color='gray', linestyle='--', label='Laatste Datum')
            plt.title('Gemiddelde Temperatuur Voorspelling')
            plt.xlabel('Datum')
            plt.ylabel('Temperatuur (°C)')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
            st.pyplot(fig)
        
        elif prediction_option == "Gecombineerde Voorspelling":
            fig, ax1 = plt.subplots(figsize=(12, 6))
        
            # Linker y-as (Aantal Ritten)
            ax1.plot(merged_df['Date'], merged_df['Number of Rides'], label='Aantal Ritten (Historisch)', color='blue')
            if toggle_predictions:
                ax1.plot(predictions_df['Date'], predictions_df['Predicted Number of Rides'], label='Aantal Ritten (Voorspelling)', color='orange', linestyle='--')
            ax1.set_xlabel('Datum')
            ax1.set_ylabel('Aantal Ritten', color='blue')
        
            # Rechter y-as (Temperatuur)
            ax2 = ax1.twinx()
            ax2.plot(merged_df['Date'], merged_df['tavg'], label='Gemiddelde Temperatuur (Historisch)', color='green')
            if toggle_predictions:
                ax2.plot(predictions_df['Date'], predictions_df['Predicted Average Temperature (tavg)'], label='Temperatuur (Voorspelling)', color='red', linestyle='--')
            ax2.set_ylabel('Temperatuur (°C)', color='green')
        
            plt.title('Gecombineerde Voorspelling: Aantal Ritten en Temperatuur')
            plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
            st.pyplot(fig)

    with tab_selection[3]:
        st.subheader("🗺️ Kaart")
        
        # Dropdown menu om te kiezen tussen beginstations en eindstations
        station_type = st.selectbox("Kies type stations:", ["Eindstations", "Beginstations"])
        
        # Verkrijg unieke datums uit de dataset
        unique_dates = pd.to_datetime(journey_df['Start Date']).dt.date.unique()
        
        # Slider om een enkele datum te filteren
        selected_date = st.slider("Selecteer een datum:", 
                                   min_value=min(unique_dates), 
                                   max_value=max(unique_dates), 
                                   value=min(unique_dates), 
                                   format="YYYY-MM-DD")
        
        # Haal de gemiddelde temperatuur van de geselecteerde datum uit journey_df
        avg_temp_row = journey_df[journey_df['Start Date'].dt.date == selected_date]
        
        if not avg_temp_row.empty:
            avg_temp = avg_temp_row['tavg'].mean()  # Neem het gemiddelde van de tavg voor de geselecteerde datum
            st.markdown(f"**Op {selected_date} was de gemiddelde temperatuur:** **{avg_temp:.2f} °C**")
        else:
            st.warning(f"Geen weerdata beschikbaar voor {selected_date}.")
        
        # Filter de dataset op basis van de geselecteerde datum
        if station_type == "Eindstations":
            data_to_plot = journey_df[pd.to_datetime(journey_df['End Date']).dt.date == selected_date]
            lat_col, long_col, name_col = 'lat_end', 'long_end', 'name_end'
        
        elif station_type == "Beginstations":
            data_to_plot = journey_df[pd.to_datetime(journey_df['Start Date']).dt.date == selected_date]
            lat_col, long_col, name_col = 'lat_start', 'long_start', 'name_start'
        
        # Tel het aantal ritten per station
        station_count = data_to_plot[name_col].value_counts().reset_index()
        station_count.columns = [name_col, 'count']
        
        # Selecteer de top 10 buurten met de meeste ritten
        top_stations = station_count.nlargest(10, 'count')[name_col].tolist()
        
        # Toevoegen van een optie voor 'Alle stations' in de sidebar
        selected_stations_option = st.sidebar.selectbox("Kies het type stations om te tonen:", 
                                                          ["Top 10 stations", "Alle stations"])
        
        if selected_stations_option == "Top 10 stations":
            selected_stations = st.sidebar.multiselect("Kies een of meer stations:", top_stations, default=top_stations)
        else:
            # Voor alle stations, haal unieke station namen op
            all_stations = station_count[name_col].tolist()
            selected_stations = st.sidebar.multiselect("Kies een of meer stations:", all_stations, default=all_stations)
        
        # Filter de data op basis van de geselecteerde stations
        data_to_plot = data_to_plot[data_to_plot[name_col].isin(selected_stations)]
        
        # Maak een folium kaart
        m = folium.Map(location=[51.5074, -0.1278], zoom_start=12)  # Coördinaten van Londen
        
        # Maak CircleMarkers aan op basis van de gefilterde data
        if not data_to_plot.empty:
            for station in selected_stations:
                station_data = data_to_plot[data_to_plot[name_col] == station]
                count = station_data[name_col].count()
                radius = max(count / 10, 5)  # Minimum radius om grote cirkels te vermijden
                
                # Bepaal de kleur op basis van het aantal ritten
                if count > 200:  # Voorbeeld drempelwaarde
                    color = 'red'
                elif count > 100:
                    color = 'orange'
                else:
                    color = 'green'
        
                row = station_data.iloc[0]  # Neem de eerste rij van het gefilterde station
                
                CircleMarker(
                    location=[row[lat_col], row[long_col]],
                    radius=radius,
                    popup=f"{station}: {count} ritten",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6
                ).add_to(m)
        
        # Weergeven van de kaart in Streamlit
        folium_static(m)
        
    with tab_selection[4]:
        st.subheader("Conclusie")
    
        st.write("""
        **Belangrijkste conclusies:**
        """)
        
        st.write("""
        📈 **Toename van fietsritten:** 
        Uit onze analyses blijkt dat het aantal fietsritten absoluut toeneemt naarmate de temperatuur stijgt. Dit geeft aan dat warmer weer een positieve invloed heeft op het gebruik van openbare fietsen.
    
        🚲 **Weekendgebruik:** 
        Echter, er zijn ook momenten waarop we het tegenovergestelde effect zien, met name in het weekend. Op deze momenten lijken mensen minder gebruik te maken van de openbare fietsen. Dit kan erop wijzen dat wanneer mensen in het weekend fietsen, ze dit vaker doen met hun eigen fiets in plaats van met een openbare fiets. 
    
        🏙️ **Werkweekbehoefte:** 
        Tijdens de werkweek is er daarentegen een grotere behoefte aan fietsen, omdat mensen ze gebruiken om van station tot station te reizen.
    
        📊 **Waardevol voor beleidsmakers:** 
        Deze inzichten zijn waardevol voor beleidsmakers en kunnen helpen bij het verbeteren van fietsvoorzieningen in Londen, vooral tijdens drukke periodes zoals de werkweek.
        """)

elif dashboard_option == "Conclusie":
    st.subheader("Conclusie")
    

    st.write("""
        🌟 **Verbeteringen in ons vernieuwde dashboard:**
    """)
    st.markdown("""
    - **Gebruiksvriendelijkheid**: Het dashboard is nu veel gebruiksvriendelijker voor de gemiddelde gebruiker.
    - **Helder inzicht**: Gebruikers krijgen een duidelijk overzicht van het algehele openbare fietsgebruik in Londen. 🚲
    - **Betere conclusies**: Ons nieuwe dashboard stelt ons in staat om betere en betrouwbaardere conclusies te trekken. 📊
    """)

