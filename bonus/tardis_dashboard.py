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
from PIL import Image


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
    st.sidebar.markdown("### Ce que vous retrouverez dans cette page :")
    st.sidebar.markdown(
        "##### Les différentes étapes de la création du modèle de prédictions"
    )
    return


def display_info_about_pred():
    img1 = Image.open("ugly_dataset.png")
    img2 = Image.open("trend.png")
    img3 = Image.open("trend_cheese.png")
    img4 = Image.open("model.jpg")

    st.title("Comment fonctionne notre model de prédictions ?")
    st.write(
        "Tout commence avec une gigantesque base de données de la SNCF\nIl contient plusieurs informations sur des trajets à certaines date comme les nombre de trains programmés, la durée moyenne des trajets et surtout les retards."
    )
    st.subheader("1 - Préparer la base de donnée")
    st.write(
        "La première partie consiste à préparer la base de données.\n On entends principalement par là, la nettoyer. Car elle est rempli de fautes (nombre de trains à virgule, temps de trajets négatifs ou fautes d'orthographes).\n"
        "Comme vous pouvez le voir dans l'extrait de la base données si dessous"
    )
    st.image(img1, width=1000)
    st.write("Il faut donc choisir quelles informations/lines corriger ou supprimer.")

    st.subheader("2 - Trouver des tendances")
    st.write(
        "La seconde partie consiste à trouver les informations les plus pertinentes."
        "C'est à dire trouver des tendances et des corrélations entre les différents données et les retards des trains."
    )
    st.image(img2, width=1000)
    st.image(img3, width=1000)
    st.write(
        "L'idée est qu'à la fin on se retrouve avec environ un milliers de graphiques afin de sélectionner les informations qui serviront de paramètres à notre modèle."
    )

    st.subheader("3 - Entraîner le modèle")
    st.write(
        "Après avoir identifié quelles données serviront au modèle pour faire des prédictions, il faut entreîner ce modèle à prédire les retards.\nMais avant même de l'entraîner il faut choisir un modèle. En effet il en existe plusieurs. Certains sont meilleurs que d'autres pour prédire certaines choses."
        "\n\nDans cette dernière partie, nous allons créer un algorythme qui va automatiser le test de plusieurs modèles avec diférents paramètres."
    )
    st.image(img4, width=1000)
    st.write(
        " L'algorythme va comparer les resultats grâce à des methodes mathématiques compliquées. L'algorythme renvoie donc le meilleur modèle avec les meilleurs paramètres. Ce modèle est directement utiliser comme fonction pour prédire des retards dans la dernière section de ce dashboard."
        "\n\nNous avons donc à la fin un modèlme qui prédit un potentiel retards par rapport à un trajet et un dates."
    )
    return


def global_pred():
    global_pred_sidebar()
    display_info_about_pred()
    return


def precise_pred_sidebar():
    # bulle info
    st.sidebar.markdown("### Ce que vous retrouverez dans cette page :")
    st.sidebar.markdown(
        "##### l'outil de prédiction de retards sur un trajet en particulier."
        "\n\nSélectionnez deux gare, une date de départ et appuyez sur \"valider\"."
    )
    # init les gares
    gares = [
        "PARIS MONTPARNASSE",
        "QUIMPER",
        "ST MALO",
        "ST PIERRE DES CORPS",
        "STRASBOURG",
        "PARIS NORD",
        "LYON PART DIEU",
        "TOURCOING",
        "NANTES",
        "PARIS VAUGIRARD",
        "BORDEAUX ST JEAN",
        "PARIS LYON",
        "MARNE LA VALLEE",
        "CHAMBERY CHALLES LES EAUX",
        "MARSEILLE ST CHARLES",
        "FRANCFORT",
        "ZURICH",
        "ANGOULEME",
        "POITIERS",
        "TOURS",
        "METZ",
        "REIMS",
        "PARIS EST",
        "DOUAI",
        "MULHOUSE VILLE",
        "VALENCE ALIXAN TGV",
        "STUTTGART",
        "BARCELONA",
        "ANGERS SAINT LAUD",
        "LAVAL",
        "NANCY",
        "LILLE",
        "GRENOBLE",
        "LE CREUSOT MONTCEAU MONTCHANIN",
        "MACON LOCHE",
        "NIMES",
        "ITALIE",
        "RENNES",
        "BREST",
        "LA ROCHELLE VILLE",
        "LE MANS",
        "VANNES",
        "DUNKERQUE",
        "AVIGNON TGV",
        "BELLEGARDE (AIN)",
        "BESANCON FRANCHE COMTE TGV",
        "DIJON VILLE",
        "MONTPELLIER",
        "MADRID",
        "ARRAS",
        "AIX EN PROVENCE TGV",
        "ANNECY",
        "NICE VILLE",
        "SAINT ETIENNE CHATEAUCREUX",
        "TOULON",
        "GENEVE",
        "PERPIGNAN",
        "LAUSANNE",
        "TOULOUSE MATABIAU",
    ]
    gares.sort()
    # choix des gares
    st.sidebar.markdown("### Où ?")
    gare_depart = st.sidebar.selectbox("Gare de départ", gares)
    gare_arrivee = st.sidebar.selectbox("Gare d'arrivée", gares)
    # choix de la date
    st.sidebar.markdown("### Quand ?")
    date_debut = st.sidebar.date_input("Date de début", min_value=datetime.today())
    # choix du service
    st.sidebar.markdown("### Service ?")
    service = st.sidebar.selectbox("Service", ["National", "International"])
    # vérifie si ya valid_journey dans la sesion actuel
    if "valid_journey" not in st.session_state:
        st.session_state.valid_journey = False
    # si il appuie sur valider
    bloc_height = 250
    if st.sidebar.button("Valider"):
        # les gares c pas les même
        if gare_depart == gare_arrivee:
            st.sidebar.error(
                "La gare de départ et la gare d'arrivée ne peuvent pas être les mêmes."
            )
            st.session_state.valid_journey = False
        # on set tout pour plus tard
        else:
            st.session_state.valid_journey = True
            st.session_state.gare_depart = gare_depart
            st.session_state.gare_arrivee = gare_arrivee
            st.session_state.date_debut = date_debut
            st.session_state.service = service
    # si le trajet est bon
    if st.session_state.valid_journey:
        # aff titre
        html_title = """
        <div style='font-size:40px; font-weight:bold; overflow: hidden; text-overflow: ellipsis;'>
            Information sur votre trajet :
        </div>
        """
        components.html(html_title, height=55)
        # aff les box avec gare et date
        date_str = (
            f"Vous partez le : {st.session_state.date_debut.strftime('%d/%m/%Y')}"
        )
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
        if (
            len(st.session_state.gare_arrivee) > 20
            or len(st.session_state.gare_depart) > 20
        ):
            bloc_height = 300
        components.html(html_code, height=bloc_height)
        # fais la map
        coord_depart = get_coordinates(st.session_state.gare_depart)
        coord_arrivee = get_coordinates(st.session_state.gare_arrivee)
        if coord_depart and coord_arrivee:
            mid_lat = (coord_depart[0] + coord_arrivee[0]) / 2
            mid_lon = (coord_depart[1] + coord_arrivee[1]) / 2

            m = folium.Map(location=[mid_lat, mid_lon], zoom_start=6)
            folium.Marker(
                coord_depart,
                tooltip="Départ",
                popup=st.session_state.gare_depart,
                icon=folium.Icon(color="green"),
            ).add_to(m)
            folium.Marker(
                coord_arrivee,
                tooltip="Arrivée",
                popup=st.session_state.gare_arrivee,
                icon=folium.Icon(color="red"),
            ).add_to(m)
            folium.PolyLine(
                [coord_depart, coord_arrivee], color="blue", weight=4
            ).add_to(m)
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
    pred1_min, pred1_sec = transform_decimal_into_min(
        predict_delay(
            str(st.session_state.date_debut),
            st.session_state.gare_depart,
            st.session_state.gare_arrivee,
            st.session_state.service,
            "Average delay of late trains at departure",
            df,
            result_df,
        )
    )
    pred2_min, pred2_sec = transform_decimal_into_min(
        predict_delay(
            str(st.session_state.date_debut),
            st.session_state.gare_depart,
            st.session_state.gare_arrivee,
            st.session_state.service,
            "Average delay of all trains at departure",
            df,
            result_df,
        )
    )
    pred3_min, pred3_sec = transform_decimal_into_min(
        predict_delay(
            str(st.session_state.date_debut),
            st.session_state.gare_depart,
            st.session_state.gare_arrivee,
            st.session_state.service,
            "Average delay of late trains at arrival",
            df,
            result_df,
        )
    )
    pred4_min, pred4_sec = transform_decimal_into_min(
        predict_delay(
            str(st.session_state.date_debut),
            st.session_state.gare_depart,
            st.session_state.gare_arrivee,
            st.session_state.service,
            "Average delay of all trains at arrival",
            df,
            result_df,
        )
    )

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
    precise_pred_sidebar()
    df = pd.read_csv("cleaned_dataset.csv")
    result_df = pd.read_csv("model_results.csv")

    # si le trajet est bon on aff
    if st.session_state.valid_journey:
        line_df = df[
            (df["Departure station"] == st.session_state.gare_depart)
            & (df["Arrival station"] == st.session_state.gare_arrivee)
        ]
        if line_df.empty:
            st.error(
                "Nous n'avons pas asez d'information sur ce trajet ou n'existe pas"
            )
            return
        print_prediction(st, df, result_df)
    return


def prediction():
    page = st.sidebar.selectbox(
        "Quelle type de prediction voulez vous ?",
        ("Notre model de prédiction", "Prédiction précise"),
    )
    if page == "Prédiction précise":
        precise_pred()
    elif page == "Notre model de prédiction":
        global_pred()
    return


def graph_average_journey_time(df):
    plt.figure(figsize=(12, 3.5))
    plt.hist(df["Average journey time"].dropna(), bins=30, edgecolor="black")
    plt.title("Durée moyenne des trajets")
    plt.xlabel("Durée (en minutes)")
    plt.ylabel("Nombre de trains")
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()
    return


def graph_top_thirty_arrival(df):
    departure_trains = df.groupby("Arrival station")["Number of scheduled trains"].sum()
    plt.figure(figsize=(14, 5))
    departure_trains.sort_values(ascending=False).head(30).plot(
        kind="bar", color="orange"
    )
    plt.title("Top 30 des gares les plus desservies à l'arrivée")
    plt.xlabel("Gare d'arrivée")
    plt.ylabel("Nombre total de trains")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return


def graph_top_thirty_departure(df):
    departure_trains = df.groupby("Departure station")[
        "Number of scheduled trains"
    ].sum()
    plt.figure(figsize=(14, 5))
    departure_trains.sort_values(ascending=False).head(30).plot(
        kind="bar", color="orange"
    )
    plt.title("Top 30 des gares les plus desservies au départ")
    plt.xlabel("Gare de départ")
    plt.ylabel("Nombre total de trains")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return


def graph_trains_delay_by_category(df):
    data_to_plot = [
        df["Number of trains delayed > 15min"].dropna(),
        df["Number of trains delayed > 30min"].dropna(),
        df["Number of trains delayed > 60min"].dropna(),
    ]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data_to_plot, tick_labels=[">15min", ">30min", ">60min"])
    plt.title("Distribution des retards par catégorie")
    plt.ylabel("Nombre de trains")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return


def graph_schedule_train_vs_arrival_delay(df):
    plt.figure(figsize=(10, 3))
    plt.scatter(
        df["Number of scheduled trains"],
        df["Number of trains delayed at arrival"],
        alpha=0.6,
    )
    plt.title("Nombre de trains prévus vs retards à l'arrivée")
    plt.xlabel("Trains prévus")
    plt.ylabel("Retards à l'arrivée")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return


def graph_schedule_train_vs_departure_delay(df):
    plt.figure(figsize=(10, 3))
    plt.scatter(
        df["Number of scheduled trains"],
        df["Number of trains delayed at departure"],
        alpha=0.6,
    )
    plt.title("Nombre de trains prévus vs retards au départ")
    plt.xlabel("Trains prévus")
    plt.ylabel("Retards au départ")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return


def graph_cause_percentage_per_month(df, cause_cols):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.strftime("%B")

    cause_columns = [col for col in df.columns if col.startswith("Pct delay due to")]
    grouped = df.groupby("Month")[cause_columns].mean()
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    grouped = grouped.reindex([m for m in month_order if m in grouped.index])

    plt.figure(figsize=(12, 4))
    bottom = None
    for col in cause_columns:
        values = grouped[col]
        plt.bar(grouped.index, values, bottom=bottom, label=col)
        if bottom is None:
            bottom = values
        else:
            bottom += values

    plt.title("Pourcentage de retards par cause et par mois")
    plt.ylabel("Pourcentage (%)")
    plt.xlabel("Mois")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()
    return


def graph_late_delay_departure(df):
    if "Date" in df.columns:
        df_time = df.set_index("Date")
        weekly_delay = (
            df_time["Average delay of late trains at departure"].resample("W").mean()
        )
        plt.figure(figsize=(12, 3.5))
        plt.plot(weekly_delay.index, weekly_delay.values, marker="o", linestyle="-")
        plt.title("Retard moyen des trains en retard au départ (par semaine)")
        plt.ylabel("Retard (en minutes)")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()
    return


def graph_late_delay_arrival(df):
    if "Date" in df.columns:
        df_time = df.set_index("Date")
        weekly_delay = (
            df_time["Average delay of late trains at arrival"].resample("W").mean()
        )
        plt.figure(figsize=(12, 3.5))
        plt.plot(weekly_delay.index, weekly_delay.values, marker="o", linestyle="-")
        plt.title("Retard moyen des trains en retard à l'arrivée (par semaine)")
        plt.ylabel("Retard (en minutes)")
        plt.grid(True)
        st.pyplot(plt)
        plt.clf()
    return


def graph_cause_evolution(df, cause_cols):
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.strftime("%B")

    df.groupby(df["Month"])[cause_cols].mean().plot()
    plt.title("Évolution des causes de retards dans le temps")
    plt.xlabel("Mois")
    plt.ylabel("% de retard")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(plt)
    plt.clf()
    return


def graph_delay_cause(df, cause_cols):
    mean_causes = df[cause_cols].mean()
    plt.figure(figsize=(12, 12))
    plt.pie(mean_causes, labels=mean_causes.index, autopct="%1.1f%%", startangle=140)
    plt.title("Répartition moyenne des causes de retards")
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()
    return


def display_global_info_graph():
    df = pd.read_csv("cleaned_dataset.csv")
    cause_cols = [
        "Pct delay due to external causes",
        "Pct delay due to infrastructure",
        "Pct delay due to traffic management",
        "Pct delay due to rolling stock",
        "Pct delay due to station management and equipment reuse",
        "Pct delay due to passenger handling (crowding, disabled persons, connections)",
    ]

    st.title("Quelques informations :")
    st.header("Statistiques sur les causes de retards :")
    with st.expander("Répartition moyenne des causes de retards"):
        graph_delay_cause(df, cause_cols)
    with st.expander("Évolution des causes de retards dans le temps"):
        graph_cause_evolution(df, cause_cols)
    with st.expander("Pourcentage de retards par cause et par mois"):
        graph_cause_percentage_per_month(df, cause_cols)

    st.header("Statistiques sur les retards :")
    with st.expander("Retard moyen à l'arrivée des trains en retard (par semaine)"):
        graph_late_delay_arrival(df)
    with st.expander("Retard moyen au départ des trains en retard (par semaine)"):
        graph_late_delay_departure(df)
    with st.expander("Trains prévus vs retards au départ"):
        graph_schedule_train_vs_departure_delay(df)
    with st.expander("Trains prévus vs retards à l'arrivée"):
        graph_schedule_train_vs_arrival_delay(df)
    with st.expander("Distribution des retards par catégorie"):
        graph_trains_delay_by_category(df)

    st.header("Statistiques diverses :")
    with st.expander("Top 30 des gares les plus desservies au départ"):
        graph_top_thirty_departure(df)
    with st.expander("Top 30 des gares les plus desservies à l'arrivée"):
        graph_top_thirty_arrival(df)
    with st.expander("Durée moyenne des trajets"):
        graph_average_journey_time(df)
    return


def global_info_sidebar():
    st.sidebar.markdown("### Ce que vous retrouverez dans cette page :")
    st.sidebar.markdown(
        "##### Plusieurs graphiques sur des statisques concernant les retards des trains de la SNCF."
    )
    return


def global_info():
    global_info_sidebar()
    display_global_info_graph()
    return


def precise_graph_average_delay_at_arrival(monthly_avg_arrival_delay):
    plt.plot(
        monthly_avg_arrival_delay.index,
        monthly_avg_arrival_delay.values,
        label="Retard moyen",
        marker="o",
        color="purple",
    )
    plt.title("Retard moyen à l'arrivée")
    plt.ylabel("Retard (en minutes)")
    plt.xticks(monthly_avg_arrival_delay.index)
    plt.tick_params(axis="x", rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    st.pyplot(plt)
    plt.clf()
    return


def precise_graph_average_delay_at_departure(monthly_avg_departure_delay):
    plt.plot(
        monthly_avg_departure_delay.index,
        monthly_avg_departure_delay.values,
        label="Retard moyen",
        marker="o",
        color="purple",
    )
    plt.title("Retard moyen au départ")
    plt.ylabel("Retard (en minutes)")
    plt.xticks(monthly_avg_departure_delay.index)
    plt.tick_params(axis="x", rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    st.pyplot(plt)
    plt.clf()
    return


def precise_graph_different_cause_delay(mean_causes):
    plt.barh(mean_causes.index, mean_causes.values, color="salmon")
    plt.title("Répartition des différentes causes de retard")
    plt.xlabel("Pourcentage moyen (%)")
    st.pyplot(plt)
    plt.clf()
    return


def precise_graph_nb_long_delay(monthly_delay_15, monthly_delay_30, monthly_delay_60):
    plt.plot(
        monthly_delay_15.index,
        monthly_delay_15.values,
        label=">15 min",
        marker="o",
        color="orange",
    )
    plt.plot(
        monthly_delay_30.index,
        monthly_delay_30.values,
        label=">30 min",
        marker="o",
        color="red",
    )
    plt.plot(
        monthly_delay_60.index,
        monthly_delay_60.values,
        label=">60 min",
        marker="o",
        color="darkred",
    )
    plt.title("Nombre de retards importants par mois")
    plt.ylabel("Nombre de trains")
    plt.xticks(monthly_delay_15.index)
    plt.tick_params(axis="x", rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(plt)
    plt.clf()
    return


def precise_graph_average_journey_time(monthly_journey_time):
    plt.plot(
        monthly_journey_time.index,
        monthly_journey_time.values,
        label="Temps de trajet",
        marker="o",
        color="royalblue",
    )
    plt.title("Temps de trajet moyen")
    plt.ylabel("Temps de trajet (en minutes)")
    plt.xticks(monthly_journey_time.index)
    plt.tick_params(axis="x", rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    st.pyplot(plt)
    plt.clf()
    return


def plot_line_dashboard(departure_station, arrival_station):
    df = pd.read_csv("cleaned_dataset.csv")

    line_df = df[
        (df["Departure station"] == departure_station)
        & (df["Arrival station"] == arrival_station)
    ]
    if line_df.empty:
        return 84

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    monthly_journey_time = (
        line_df.groupby("Month")["Average journey time"].mean().reindex(month_order)
    )
    monthly_delay_15 = (
        line_df.groupby("Month")["Number of trains delayed > 15min"]
        .sum()
        .reindex(month_order)
    )
    monthly_delay_30 = (
        line_df.groupby("Month")["Number of trains delayed > 30min"]
        .sum()
        .reindex(month_order)
    )
    monthly_delay_60 = (
        line_df.groupby("Month")["Number of trains delayed > 60min"]
        .sum()
        .reindex(month_order)
    )
    monthly_avg_arrival_delay = (
        line_df.groupby("Month")["Average delay of all trains at arrival"]
        .mean()
        .reindex(month_order)
    )
    monthly_avg_departure_delay = (
        line_df.groupby("Month")["Average delay of all trains at departure"]
        .mean()
        .reindex(month_order)
    )

    causes_cols = [
        "Pct delay due to external causes",
        "Pct delay due to infrastructure",
        "Pct delay due to traffic management",
        "Pct delay due to rolling stock",
        "Pct delay due to station management and equipment reuse",
        "Pct delay due to passenger handling (crowding, disabled persons, connections)",
    ]
    mean_causes = line_df[causes_cols].mean().dropna().sort_values()

    st.title("Quelques informations :")
    st.header("Statistiques sur les causes de retards :")
    with st.expander("Répartition moyenne des causes de retard"):
        precise_graph_different_cause_delay(mean_causes)

    st.header("Statistiques sur les retards :")
    with st.expander("Retard moyen au départ"):
        precise_graph_average_delay_at_departure(monthly_avg_departure_delay)
    with st.expander("Retard moyen à l'arrivée"):
        precise_graph_average_delay_at_arrival(monthly_avg_arrival_delay)

    st.header("Statistiques diverses :")
    with st.expander("Temps de trajet moyen"):
        precise_graph_average_journey_time(monthly_journey_time)
    with st.expander("Nombre de retards importants par mois"):
        precise_graph_nb_long_delay(
            monthly_delay_15, monthly_delay_30, monthly_delay_60
        )
    return 0


def precise_info_sidebar():
    # bulle info
    st.sidebar.markdown("### Ce que vous retrouverez dans cette page :")
    st.sidebar.markdown(
        "##### Des informations et des graphiques sur un trajet spécifique."
        "\n\nSélectionnez deux gare, une date de départ et appuyez sur valider."
    )
    # init les gares
    gares = [
        "PARIS MONTPARNASSE",
        "QUIMPER",
        "ST MALO",
        "ST PIERRE DES CORPS",
        "STRASBOURG",
        "PARIS NORD",
        "LYON PART DIEU",
        "TOURCOING",
        "NANTES",
        "PARIS VAUGIRARD",
        "BORDEAUX ST JEAN",
        "PARIS LYON",
        "MARNE LA VALLEE",
        "CHAMBERY CHALLES LES EAUX",
        "MARSEILLE ST CHARLES",
        "FRANCFORT",
        "ZURICH",
        "ANGOULEME",
        "POITIERS",
        "TOURS",
        "METZ",
        "REIMS",
        "PARIS EST",
        "DOUAI",
        "MULHOUSE VILLE",
        "VALENCE ALIXAN TGV",
        "STUTTGART",
        "BARCELONA",
        "ANGERS SAINT LAUD",
        "LAVAL",
        "NANCY",
        "LILLE",
        "GRENOBLE",
        "LE CREUSOT MONTCEAU MONTCHANIN",
        "MACON LOCHE",
        "NIMES",
        "ITALIE",
        "RENNES",
        "BREST",
        "LA ROCHELLE VILLE",
        "LE MANS",
        "VANNES",
        "DUNKERQUE",
        "AVIGNON TGV",
        "BELLEGARDE (AIN)",
        "BESANCON FRANCHE COMTE TGV",
        "DIJON VILLE",
        "MONTPELLIER",
        "MADRID",
        "ARRAS",
        "AIX EN PROVENCE TGV",
        "ANNECY",
        "NICE VILLE",
        "SAINT ETIENNE CHATEAUCREUX",
        "TOULON",
        "GENEVE",
        "PERPIGNAN",
        "LAUSANNE",
        "TOULOUSE MATABIAU",
    ]
    gares.sort()
    # choix des gares
    st.sidebar.markdown("### Où ?")
    gare_depart = st.sidebar.selectbox("Gare de départ", gares)
    gare_arrivee = st.sidebar.selectbox("Gare d'arrivée", gares)
    # choix de la date
    st.sidebar.markdown("### Quand ?")
    date_debut = st.sidebar.date_input("Date de début", min_value=datetime.today())
    # vérifie si ya valid_journey dans la sesion actuel
    if "valid_journey" not in st.session_state:
        st.session_state.valid_journey = False
    # si il appuie sur valider
    bloc_height = 250
    if st.sidebar.button("Valider"):
        # les gares c pas les même
        if gare_depart == gare_arrivee:
            st.sidebar.error(
                "La gare de départ et la gare d'arrivée ne peuvent pas être les mêmes."
            )
            st.session_state.valid_journey = False
        # on set tout pour plus tard
        else:
            st.session_state.valid_journey = True
            st.session_state.gare_depart = gare_depart
            st.session_state.gare_arrivee = gare_arrivee
            st.session_state.date_debut = date_debut
    # si le trajet est bon
    if st.session_state.valid_journey:
        # aff titre
        html_title = """
        <div style='font-size:40px; font-weight:bold; overflow: hidden; text-overflow: ellipsis;'>
            Information sur votre trajet :
        </div>
        """
        components.html(html_title, height=55)
        # aff les box avec gare et date
        date_str = (
            f"Vous partez le : {st.session_state.date_debut.strftime('%d/%m/%Y')}"
        )
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
        if (
            len(st.session_state.gare_arrivee) > 20
            or len(st.session_state.gare_depart) > 20
        ):
            bloc_height = 300
        components.html(html_code, height=bloc_height)
        # fais la map
        coord_depart = get_coordinates(st.session_state.gare_depart)
        coord_arrivee = get_coordinates(st.session_state.gare_arrivee)
        if coord_depart and coord_arrivee:
            mid_lat = (coord_depart[0] + coord_arrivee[0]) / 2
            mid_lon = (coord_depart[1] + coord_arrivee[1]) / 2

            m = folium.Map(location=[mid_lat, mid_lon], zoom_start=6)
            folium.Marker(
                coord_depart,
                tooltip="Départ",
                popup=st.session_state.gare_depart,
                icon=folium.Icon(color="green"),
            ).add_to(m)
            folium.Marker(
                coord_arrivee,
                tooltip="Arrivée",
                popup=st.session_state.gare_arrivee,
                icon=folium.Icon(color="red"),
            ).add_to(m)
            folium.PolyLine(
                [coord_depart, coord_arrivee], color="blue", weight=4
            ).add_to(m)
            st_folium(m, width=700, height=500)
        else:
            st.error("Impossible de géolocaliser l'une des deux gares.")
        return st.session_state


def precise_info():
    precise_info_sidebar()
    # si le trajet est bon on aff
    if st.session_state.valid_journey:
        error = plot_line_dashboard(
            st.session_state.gare_depart, st.session_state.gare_arrivee
        )
        if error == 84:
            st.error(
                "Nous n'avons pas assez d'information sur ce trajet ou ce trajet n'existe pas."
            )

    return


def information():
    page = st.sidebar.selectbox(
        "Globale ou précise ?",
        ("Information globale", "Information précise"),
    )
    if page == "Information précise":
        precise_info()
    elif page == "Information globale":
        global_info()
    return


def main():
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox(
        "Que voulez vous chercher ?", ("Information", "Prediction")
    )

    if page == "Prediction":
        prediction()
    elif page == "Information":
        information()

    return


if main() == 84:
    exit(84)
