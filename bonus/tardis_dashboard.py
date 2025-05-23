#!/usr/bin/python3

import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium
import requests
from tardis_model import predict_delay




def get_coordinates(nom_gare):
    GEOAPIFY_API_KEY = "224eacac2c7746d1a06ef91cc64ec5f4"
    url = f"https://api.geoapify.com/v1/geocode/search?text=Gare {nom_gare}, France&apiKey={GEOAPIFY_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data["features"]:
            coords = data["features"][0]["geometry"]["coordinates"]
            return coords[1], coords[0]
    return None









def global_pred_sidebar():
    st.sidebar.markdown("### ce que vous retrouverez dans cette page :")
    st.sidebar.markdown("##### On doit écrire des truc ici, mais je sais pas quoi, ducoup je met des truc aléatoire , martin a de petit pied")
    return

def global_pred():
    global_pred_sidebar()
    st.write("martin")
    return

def precise_pred_sidebar():
    #bulle info
    st.sidebar.markdown("### Ce que vous retrouverez dans cette page :")
    st.sidebar.markdown("##### On doit écrire des petit pied")
    #init les gares
    gares = [
        "PARIS MONTPARNASSE", "QUIMPER", "ST MALO", "ST PIERRE DES CORPS", "STRASBOURG", "PARIS NORD", "LYON PART DIEU", "TOURCOING",
        "NANTES", "PARIS VAUGIRARD", "BORDEAUX ST JEAN", "PARIS LYON", "MARNE LA VALLEE", "CHAMBERY CHALLES LES EAUX", "MARSEILLE ST CHARLES", "FRANCFORT", "ZURICH", "ANGOULEME", "POITIERS", "TOURS", "METZ", "REIMS", "PARIS EST", "DOUAI", "MULHOUSE VILLE", "VALENCE ALIXAN TGV", "STUTTGART", "BARCELONA",
        "ANGERS SAINT LAUD", "LAVAL", "NANCY", "LILLE", "GRENOBLE", "LE CREUSOT MONTCEAU MONTCHANIN", "MACON LOCHE",
        "NIMES", "ITALIE", "RENNES", "BREST", "LA ROCHELLE VILLE", "LE MANS", "VANNES", "DUNKERQUE", "AVIGNON TGV", "BELLEGARDE (AIN)", "BESANCON FRANCHE COMTE TGV", "DIJON VILLE", "MONTPELLIER", "MADRID", "ARRAS", "AIX EN PROVENCE TGV",
        "ANNECY", "NICE VILLE", "SAINT ETIENNE CHATEAUCREUX", "TOULON", "GENEVE", "PERPIGNAN", "LAUSANNE", "TOULOUSE MATABIAU"
    ]
    gares.sort()
    #choix des gares
    st.sidebar.markdown("### Où ?")
    gare_depart = st.sidebar.selectbox("Gare de départ", gares)
    gare_arrivee = st.sidebar.selectbox("Gare d'arrivée", gares)
    #choix de la date
    st.sidebar.markdown("### Quand ?")
    date_debut = st.sidebar.date_input("Date de début", min_value=datetime.today())
    #vérifie si ya valid_journey dans la sesion actuel
    if "valid_journey" not in st.session_state:
        st.session_state.valid_journey = False
    #si il appuie sur valider
    bloc_height = 250
    if st.sidebar.button("Valider"):
        #les gares c pas les même
        if gare_depart == gare_arrivee:
            st.sidebar.error("La gare de départ et la gare d'arrivée ne peuvent pas être les mêmes.")
            st.session_state.valid_journey = False
        #on set tout pour plus tard
        else:
            st.session_state.valid_journey = True
            st.session_state.gare_depart = gare_depart
            st.session_state.gare_arrivee = gare_arrivee
            st.session_state.date_debut = date_debut
    #si le trajet est bon
    if st.session_state.valid_journey:
        #aff titre
        html_title = """
        <div style='font-size:40px; font-weight:bold; overflow: hidden; text-overflow: ellipsis;'>
            Information sur votre trajet :
        </div>
        """
        components.html(html_title, height=55)
        #aff les box avec gare et date
        date_str = f"Vous partez le : {st.session_state.date_debut.strftime('%d/%m/%Y')}"
        html_code = f"""
        <div style='font-size:24px; font-weight:bold; padding: 10px; margin-bottom: 10px;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px; gap: 20px;'>
                <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; width: 48%; text-align: center;'>
                    <strong>Départ :</strong><br>{st.session_state.gare_depart}
                </div>
                <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; width: 48%; text-align: center;'>
                    <strong>Arrivée :</strong><br>{st.session_state.gare_arrivee}
                </div>
            </div>
            <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; text-align: center;'>
                {date_str}
            </div>
        </div>
        """
        if len(st.session_state.gare_arrivee) > 20 or len(st.session_state.gare_depart) > 20:
            bloc_height = 300
        components.html(html_code, height=bloc_height)
        #fais la map
        coord_depart = get_coordinates(st.session_state.gare_depart)
        coord_arrivee = get_coordinates(st.session_state.gare_arrivee)
        if coord_depart and coord_arrivee:
            mid_lat = (coord_depart[0] + coord_arrivee[0]) / 2
            mid_lon = (coord_depart[1] + coord_arrivee[1]) / 2

            m = folium.Map(location=[mid_lat, mid_lon], zoom_start=6)
            folium.Marker(coord_depart, tooltip="Départ", popup=st.session_state.gare_depart, icon=folium.Icon(color='green')).add_to(m)
            folium.Marker(coord_arrivee, tooltip="Arrivée", popup=st.session_state.gare_arrivee, icon=folium.Icon(color='red')).add_to(m)
            folium.PolyLine([coord_depart, coord_arrivee], color="blue", weight=4).add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.error("Impossible de géolocaliser l'une des deux gares.")
        return st.session_state





# Average delay of late trains at departure", df, results_df):.2f} minutes")
# Average delay of all trains at departure", df, results_df):.2f} minutes")
# Average delay of late trains at arrival", df, results_df):.2f} minutes")
# Average delay of all trains at arrival", df, results_df):.2f} minutes")

def transform_decimal_into_min(pred):
    pred_min = int(pred)
    pred_sec = round((pred - pred_min) * 60)
    return pred_min, pred_sec

def print_prediction(st, df, result_df):
    pred1_min, pred1_sec = transform_decimal_into_min(predict_delay(str(st.session_state.date_debut),
        st.session_state.gare_depart, st.session_state.gare_arrivee,
        "Average delay of late trains at departure", df, result_df))
    pred2_min, pred2_sec = transform_decimal_into_min(predict_delay(str(st.session_state.date_debut),
        st.session_state.gare_depart, st.session_state.gare_arrivee,
        "Average delay of all trains at departure", df, result_df))
    pred3_min, pred3_sec = transform_decimal_into_min(predict_delay(str(st.session_state.date_debut),
        st.session_state.gare_depart, st.session_state.gare_arrivee,
        "Average delay of late trains at arrival", df, result_df))
    pred4_min, pred4_sec = transform_decimal_into_min(predict_delay(str(st.session_state.date_debut),
        st.session_state.gare_depart, st.session_state.gare_arrivee,
        "Average delay of all trains at arrival", df, result_df))


    html_code = f"""
    <div style='font-size:24px; font-weight:bold; padding: 10px; margin-bottom: 10px;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 10px; gap: 20px;'>
            <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; width: 48%; text-align: center;'>
                <strong>Retard des trains en retard au départ:</strong><br>{pred1_min}min {pred1_sec}s
            </div>
            <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; width: 48%; text-align: center;'>
                <strong>Retard des train au départ :</strong><br>{pred2_min}min {pred2_sec}s
            </div>
        </div>
        <div style='display: flex; justify-content: space-between; margin-bottom: 10px; gap: 20px;'>
            <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; width: 48%; text-align: center;'>
                <strong>Retard des trains en retard a l'arrivé :</strong><br>{pred3_min}min {pred3_sec}s
            </div>
            <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; width: 48%; text-align: center;'>
                <strong>Retard des trains a l'arrivé :</strong><br>{pred4_min}min {pred4_sec}s
            </div>
        </div>
    </div>
    """
    components.html(html_code, height=400)
    return


def precise_pred():
    df = pd.read_csv('cleaned_dataset.csv')
    result_df = pd.read_csv('model_results.csv')

    #si le trajet est bon on aff
    if st.session_state.valid_journey:
        line_df = df[(df['Departure station'] == st.session_state.gare_depart) &
                 (df['Arrival station'] == st.session_state.gare_arrivee)]
        if line_df.empty:
            st.error("Nous n'avons pas asez d'information sur ce trajet ou n'existe pas")
            return
        print_prediction(st, df, result_df)
    return

def prediction():
    page = st.sidebar.selectbox(
        'Quelle type de prediction voulez vous ?',
        ('Notre model de prédiction', 'Prédiction précise')
    )
    if page == "Prédiction précise":
        precise_pred()
    elif page == "Notre model de prédiction":
        global_pred()
    return












































































































def graph_average_journey_time(df):
    plt.figure(figsize=(12, 3.5))
    plt.hist(df['Average journey time'].dropna(), bins=30, edgecolor='black')
    plt.title("Average journey time")
    plt.xlabel("Time (in minutes)")
    plt.ylabel("Number of trains")
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()
    return

def graph_top_thirty_arrival(df):
    departure_trains = df.groupby('Arrival station')['Number of scheduled trains'].sum()
    plt.figure(figsize=(14, 5))
    departure_trains.sort_values(ascending=False).head(30).plot(kind='bar', color='orange')
    plt.title('Top 30 of the most total number of trains by arrival station')
    plt.xlabel('Arrival station')
    plt.ylabel('Total number of trains')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return

def graph_top_thirty_departure(df):
    departure_trains = df.groupby('Departure station')['Number of scheduled trains'].sum()
    plt.figure(figsize=(14, 5))
    departure_trains.sort_values(ascending=False).head(30).plot(kind='bar', color='orange')
    plt.title('Top 30 of the most total number of trains by departure station')
    plt.xlabel('Departure station')
    plt.ylabel('Total number of trains')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return

def graph_trains_delay_by_category(df):
    # get each values from >15, >30 and >60
    data_to_plot = [
        df['Number of trains delayed > 15min'].dropna(),
        df['Number of trains delayed > 30min'].dropna(),
        df['Number of trains delayed > 60min'].dropna()
    ]
    
    # convert all this data to display a boxplot
    plt.figure(figsize=(8, 5))
    plt.boxplot(data_to_plot, tick_labels=['>15min', '>30min', '>60min'])
    plt.title("Distribution of Train Delays by Category")
    plt.ylabel("Number of Trains")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return

def graph_schedule_train_vs_arrival_delay(df):
    plt.figure(figsize=(10, 3))
    plt.scatter(df['Number of scheduled trains'], df['Number of trains delayed at arrival'], alpha=0.6)
    plt.title('Scheduled Trains vs Arrival Delays')
    plt.xlabel('Scheduled Trains')
    plt.ylabel('Arrival Delays')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return

def graph_schedule_train_vs_departure_delay(df):
    plt.figure(figsize=(10, 3))
    plt.scatter(df['Number of scheduled trains'], df['Number of trains delayed at departure'], alpha=0.6)
    plt.title('Scheduled Trains vs Departure Delays')
    plt.xlabel('Scheduled Trains')
    plt.ylabel('Departure Delays')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return

def graph_cause_percentage_per_month(df, cause_cols):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = df['Date'].dt.strftime('%B')
    
    # Get the columns we need for the graph that contains delay, groupby the columns and reorder the month so it is
    # not in alphabetical order
    
    cause_columns = [col for col in df.columns if col.startswith("Pct delay due to")]
    grouped = df.groupby('Month')[cause_columns].mean()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Group the index by month
    grouped = grouped.reindex([m for m in month_order if m in grouped.index])
    
    # Call matplotlib to render the graph
    plt.figure(figsize=(12, 4))
    bottom = None
    for col in cause_columns:
        values = grouped[col]
        plt.bar(grouped.index, values, bottom=bottom, label=col)
        if bottom is None:
            bottom = values
        else:
            bottom += values
    
    # Others informations that we need (title, x and y legend)
    plt.title("Percentage of delay by differents reasons by month")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Month")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()
    return

def graph_late_delay_departure(df):
    if 'Date' in df.columns:
        df_time = df.set_index('Date')
        weekly_delay = df_time['Average delay of late trains at departure'].resample('W').mean()
        plt.figure(figsize=(12, 3.5))
        plt.plot(weekly_delay.index, weekly_delay.values, marker='o', linestyle='-')
        plt.title("Average delay of late trains at arrival (by years)")
        plt.ylabel("Delay (in minutes)")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()
    return

def graph_late_delay_arrival(df):
    if 'Date' in df.columns:
        df_time = df.set_index('Date')
        weekly_delay = df_time['Average delay of late trains at arrival'].resample('W').mean()
        plt.figure(figsize=(12, 3.5))
        plt.plot(weekly_delay.index, weekly_delay.values, marker='o', linestyle='-')
        plt.title("Average delay of late trains at arrival (by years)")
        plt.ylabel("Delay (in minutes)")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()
    return

def graph_cause_evolution(df, cause_cols):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = df['Date'].dt.strftime('%B')
    
    df.groupby(df['Month'])[cause_cols].mean().plot()
    plt.title("Delay Causes Evolution Over Time")
    plt.xlabel("Month")
    plt.ylabel("% of Delay")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)
    plt.clf()
    return

def graph_delay_cause(df, cause_cols):
    mean_causes = df[cause_cols].mean()
    plt.figure(figsize=(12, 12))
    plt.pie(mean_causes, labels=mean_causes.index, autopct='%1.1f%%', startangle=140)
    plt.title('Average Delay Causes Distribution')
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return

def display_global_info_graph():
    df = pd.read_csv('cleaned_dataset.csv')
    cause_cols = [
        'Pct delay due to external causes',
        'Pct delay due to infrastructure',
        'Pct delay due to traffic management',
        'Pct delay due to rolling stock', 
        'Pct delay due to station management and equipment reuse',
        'Pct delay due to passenger handling (crowding, disabled persons, connections)'
    ]

    st.title("Quelques informations :")
    st.header("Statistique sur les causes de retards :")
    with st.expander("Average Delay Causes Distribution"):
        graph_delay_cause(df, cause_cols)
    with st.expander("Average Delay Causes Distribution"):
        graph_cause_evolution(df, cause_cols)
    with st.expander("Average Delay Causes Distribution"):
        graph_cause_percentage_per_month(df, cause_cols)

    st.header("Statistique sur les retards :")
    with st.expander("Average Delay Causes Distribution"):
        graph_late_delay_arrival(df)
    with st.expander("Average Delay Causes Distribution"):
        graph_late_delay_departure(df)
    with st.expander("Average Delay Causes Distribution"):
        graph_schedule_train_vs_departure_delay(df)
    with st.expander("Average Delay Causes Distribution"):
        graph_schedule_train_vs_arrival_delay(df)
    with st.expander("Average Delay Causes Distribution"):
        graph_trains_delay_by_category(df)

    st.header("Statistique Divers :")
    with st.expander("Average Delay Causes Distribution"):
        graph_top_thirty_departure(df)
    with st.expander("Average Delay Causes Distribution"):
        graph_top_thirty_arrival(df)
    with st.expander("Average Delay Causes Distribution"):
        graph_average_journey_time(df)
    return
















def global_info_sidebar():
    st.sidebar.markdown("### ce que vous retrouverez dans cette page :")
    st.sidebar.markdown("##### On doit écrire des truc ici, mais je sais pas quoi, ducoup je met des truc aléatoire , martin a de petit pied")
    return

def global_info():
    global_info_sidebar()
    display_global_info_graph()
    return












def precise_graph_average_delay_at_arrival(monthly_avg_arrival_delay):
    plt.plot(monthly_avg_arrival_delay.index, monthly_avg_arrival_delay.values, label="Average delay", marker='o', color='purple')
    plt.title("Average delay at arrival")
    plt.ylabel("Delay (in min)")
    plt.xticks(monthly_avg_arrival_delay.index)
    plt.tick_params(axis='x', rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    st.pyplot(plt)
    plt.clf()
    return

def precise_graph_average_delay_at_departure(monthly_avg_departure_delay):
    plt.plot(monthly_avg_departure_delay.index, monthly_avg_departure_delay.values, label="Average delay", marker='o', color='purple')
    plt.title("Average delay at departure")
    plt.ylabel("Delay (in min)")
    plt.xticks(monthly_avg_departure_delay.index)
    plt.tick_params(axis='x', rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    st.pyplot(plt)
    plt.clf()
    return

def precise_graph_different_cause_delay(mean_causes):
    plt.barh(mean_causes.index, mean_causes.values, color='salmon')
    plt.title("Distribution of the different causes of delay")
    plt.xlabel("Average percentage (%)")
    st.pyplot(plt)
    plt.clf()
    return

def precise_graph_nb_long_delay(monthly_delay_15, monthly_delay_30, monthly_delay_60):
    plt.plot(monthly_delay_15.index, monthly_delay_15.values, label=">15 min", marker='o', color='orange')
    plt.plot(monthly_delay_30.index, monthly_delay_30.values, label=">30 min", marker='o', color='red')
    plt.plot(monthly_delay_60.index, monthly_delay_60.values, label=">60 min", marker='o', color='darkred')
    plt.title("Number of long delays by month")
    plt.ylabel("Number of trains")
    plt.xticks(monthly_delay_15.index)
    plt.tick_params(axis='x', rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)
    plt.clf()
    return

def precise_graph_average_journey_time(monthly_journey_time):
    plt.plot(monthly_journey_time.index, monthly_journey_time.values, label="Journey time", marker='o', color='royalblue')
    plt.title("Average journey time")
    plt.ylabel("Journey time (in min)")
    plt.xticks(monthly_journey_time.index)
    plt.tick_params(axis='x', rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(plt)
    plt.clf()
    return

def plot_line_dashboard(departure_station, arrival_station):
    df = pd.read_csv('cleaned_dataset.csv')

    line_df = df[(df['Departure station'] == departure_station) &
                 (df['Arrival station'] == arrival_station)]
    if line_df.empty:
        return 84

    # reorder the month so its not in alphabetical order anymore
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    monthly_journey_time = line_df.groupby('Month')['Average journey time'].mean().reindex(month_order)
    monthly_delay_15 = line_df.groupby('Month')['Number of trains delayed > 15min'].sum().reindex(month_order)
    monthly_delay_30 = line_df.groupby('Month')['Number of trains delayed > 30min'].sum().reindex(month_order)
    monthly_delay_60 = line_df.groupby('Month')['Number of trains delayed > 60min'].sum().reindex(month_order)
    monthly_avg_arrival_delay = line_df.groupby('Month')['Average delay of all trains at arrival'].mean().reindex(month_order)
    monthly_avg_departure_delay = line_df.groupby('Month')['Average delay of all trains at departure'].mean().reindex(month_order)

    # colons for the reason of delay
    causes_cols = [
        'Pct delay due to external causes',
        'Pct delay due to infrastructure',
        'Pct delay due to traffic management',
        'Pct delay due to rolling stock',
        'Pct delay due to station management and equipment reuse',
        'Pct delay due to passenger handling (crowding, disabled persons, connections)'
    ]
    mean_causes = line_df[causes_cols].mean().dropna().sort_values()

    st.title("Quelques informations :")
    st.header("Statistique sur les causes de retards :")
    with st.expander("Average Delay Causes Distribution"):
        # Graph °3: Distribution of the different causes of delay
        precise_graph_different_cause_delay(mean_causes)

    st.header("Statistique sur les retards :")
    with st.expander("Average Delay Causes Distribution"):
        # Graph °4: Average delay at arrival
        precise_graph_average_delay_at_departure(monthly_avg_departure_delay)
    with st.expander("Average Delay Causes Distribution"):
        precise_graph_average_delay_at_arrival(monthly_avg_arrival_delay)

    st.header("Statistique Divers :")
    with st.expander("Average Delay Causes Distribution"):
        # Graph °1: Average journey time
        precise_graph_average_journey_time(monthly_journey_time)
    with st.expander("Average Delay Causes Distribution"):
        # Graph °2: Number of long delays by  month
        precise_graph_nb_long_delay(monthly_delay_15, monthly_delay_30, monthly_delay_60)
    return 0

def precise_info_sidebar():
    #bulle info
    st.sidebar.markdown("### Ce que vous retrouverez dans cette page :")
    st.sidebar.markdown("##### On doit écrire des petit pied")
    #init les gares
    gares = [
        "PARIS MONTPARNASSE", "QUIMPER", "ST MALO", "ST PIERRE DES CORPS", "STRASBOURG", "PARIS NORD", "LYON PART DIEU", "TOURCOING",
        "NANTES", "PARIS VAUGIRARD", "BORDEAUX ST JEAN", "PARIS LYON", "MARNE LA VALLEE", "CHAMBERY CHALLES LES EAUX", "MARSEILLE ST CHARLES", "FRANCFORT", "ZURICH", "ANGOULEME", "POITIERS", "TOURS", "METZ", "REIMS", "PARIS EST", "DOUAI", "MULHOUSE VILLE", "VALENCE ALIXAN TGV", "STUTTGART", "BARCELONA",
        "ANGERS SAINT LAUD", "LAVAL", "NANCY", "LILLE", "GRENOBLE", "LE CREUSOT MONTCEAU MONTCHANIN", "MACON LOCHE",
        "NIMES", "ITALIE", "RENNES", "BREST", "LA ROCHELLE VILLE", "LE MANS", "VANNES", "DUNKERQUE", "AVIGNON TGV", "BELLEGARDE (AIN)", "BESANCON FRANCHE COMTE TGV", "DIJON VILLE", "MONTPELLIER", "MADRID", "ARRAS", "AIX EN PROVENCE TGV",
        "ANNECY", "NICE VILLE", "SAINT ETIENNE CHATEAUCREUX", "TOULON", "GENEVE", "PERPIGNAN", "LAUSANNE", "TOULOUSE MATABIAU"
    ]
    gares.sort()
    #choix des gares
    st.sidebar.markdown("### Où ?")
    gare_depart = st.sidebar.selectbox("Gare de départ", gares)
    gare_arrivee = st.sidebar.selectbox("Gare d'arrivée", gares)
    #choix de la date
    st.sidebar.markdown("### Quand ?")
    date_debut = st.sidebar.date_input("Date de début", min_value=datetime.today())
    #vérifie si ya valid_journey dans la sesion actuel
    if "valid_journey" not in st.session_state:
        st.session_state.valid_journey = False
    #si il appuie sur valider
    bloc_height = 250
    if st.sidebar.button("Valider"):
        #les gares c pas les même
        if gare_depart == gare_arrivee:
            st.sidebar.error("La gare de départ et la gare d'arrivée ne peuvent pas être les mêmes.")
            st.session_state.valid_journey = False
        #on set tout pour plus tard
        else:
            st.session_state.valid_journey = True
            st.session_state.gare_depart = gare_depart
            st.session_state.gare_arrivee = gare_arrivee
            st.session_state.date_debut = date_debut
    #si le trajet est bon
    if st.session_state.valid_journey:
        #aff titre
        html_title = """
        <div style='font-size:40px; font-weight:bold; overflow: hidden; text-overflow: ellipsis;'>
            Information sur votre trajet :
        </div>
        """
        components.html(html_title, height=55)
        #aff les box avec gare et date
        date_str = f"Vous partez le : {st.session_state.date_debut.strftime('%d/%m/%Y')}"
        html_code = f"""
        <div style='font-size:24px; font-weight:bold; padding: 10px; margin-bottom: 10px;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 10px; gap: 20px;'>
                <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; width: 48%; text-align: center;'>
                    <strong>Départ :</strong><br>{st.session_state.gare_depart}
                </div>
                <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; width: 48%; text-align: center;'>
                    <strong>Arrivée :</strong><br>{st.session_state.gare_arrivee}
                </div>
            </div>
            <div style='background-color: #f2f2f2; padding: 20px; border-radius: 8px; text-align: center;'>
                {date_str}
            </div>
        </div>
        """
        if len(st.session_state.gare_arrivee) > 20 or len(st.session_state.gare_depart) > 20:
            bloc_height = 300
        components.html(html_code, height=bloc_height)
        #fais la map
        coord_depart = get_coordinates(st.session_state.gare_depart)
        coord_arrivee = get_coordinates(st.session_state.gare_arrivee)
        if coord_depart and coord_arrivee:
            mid_lat = (coord_depart[0] + coord_arrivee[0]) / 2
            mid_lon = (coord_depart[1] + coord_arrivee[1]) / 2

            m = folium.Map(location=[mid_lat, mid_lon], zoom_start=6)
            folium.Marker(coord_depart, tooltip="Départ", popup=st.session_state.gare_depart, icon=folium.Icon(color='green')).add_to(m)
            folium.Marker(coord_arrivee, tooltip="Arrivée", popup=st.session_state.gare_arrivee, icon=folium.Icon(color='red')).add_to(m)
            folium.PolyLine([coord_depart, coord_arrivee], color="blue", weight=4).add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.error("Impossible de géolocaliser l'une des deux gares.")
        return st.session_state


def precise_info():
    #si le trajet est bon on aff
    if st.session_state.valid_journey:
        error = plot_line_dashboard(st.session_state.gare_depart, st.session_state.gare_arrivee)
        if error == 84:
            st.error("Nous n'avons pas aasez d'information sur ce trajet")
            
    return

def information():
    page = st.sidebar.selectbox(
        'Quelle type de prediction voulez vous ?',
        ('Information global', 'Information précise')
    )
    if page == "Information précise":
        precise_info()
    elif page == "Information global":
        global_info()
    return

























def main():
    st.sidebar.title('Menu')
    page = st.sidebar.selectbox(
        'Que voulez vous chercher ?',
        ('Information', 'Prediction')
    )
    
    if page == "Prediction":
        prediction()
    elif page == "Information":
        information()

    return

if main() == 84:
    exit(84)

