# PROCESSING

Aquesta carpeta conté tots els fitxers necessaris per al processat de les dades d'entrada per a poder obtenir tota la informació necessaria per a poder mostrar les visualitzacions desitjades. El codi ha estat executat per les tres zones de les quals es té informació:

* El Canigó (fitxer d'entrada: `canigo.zip`). Zona definida per les fites `[(42.4, 2.2)- (42.6, 2.7)]`. `x` fitxers inicials de *tracks*.
* Matagalls (fitxer d'entrada `matagalls.zip`). Zona definida per les fites `[(41.8, 2.3)- (41.8, 2.5)]`. `x` fitxers inicials de *tracks*.
* Vall Ferrera (fitxer d'entrada `vallferrera.zip`). Zona definida per les fites `[(42.5, 1.2)- (42.8, 1.7)]`. `x` fitxers inicials de *tracks*.

Com podem veure, l'entrada del codi serà un fitxer *zip* que contindrà tots els *tracks* d'entrada per la zona definida. Degut a que cada carpeta conté molts *tracks*, s'ha adaptat el codi per tal de poder anar processant rutes en execucions diferents; és a dir, es va mantenint un registre de les rutes processades per tal de no processar rutes ja tractades en execucions posteriors. 

A continuació, s'explicarà tot el procés realitzat, el qual es troba dividit en tres parts: **PREPROCESSING**, **FMM-PROCESSING** i **POSTPROCESSING**. També es comentarà l'estructura de les dades d'entrada i les de sortida. 

## 1. PREPROCESSING

Aquesta primera part del processat es basa en definir totes les adreces de totes les dades i llegir totes les bases de dades per a poder-les utilitzar en un futur. Podem utilitzar aquest codi per a crear les carpetes i els fitxers en una execució inicial, o llegir totes les dades i *paths* en execucions posteriors. 

El pseudocodi (diferenciat per cada zona) és el següent:

1. Cridem la funció `extract_zip()`, la qual llegeix el fitxer *zip* inicial i passa totes les seves dades inicials a una carpeta provisional. A posteriori, tots els fitxers en format *json* (dades de rutes) es passen a una carpeta definida com `Input-Data`. Aquesta funció només s'executa si no existeix la carpeta amb els fitxers *json*. 
2. Utilitzem la funció `create_osm_network()` per a crear la xarxa de camins d'*Open Street Map*. Per a fer-ho es crear un polígon i un graf utilitzant la llibreria `osmnx` (utilitzant les fites de cada zona). Per a cada zona, es crea un fitxer en format *shapefile* per als nodes i els eixos dels camins. Anomenem la carpta amb totes les dades `OSM-Data`. La funció s'executa únicament si la carpeta no es troba creada.
3. Cridem la funció `output_structure()`, la qual crea tota l'estructura de carpetes i *dataframes* que necessitem per la sortida de dades. Entre aquestes trobem:

    * La carpeta `Output-Data`, que és on trobarem totes les altres carpetes. 
    * La carpeta `Dataframes` (dins de `Output-Data`), que conté tots els *dataframes* que esmentarem a continuació. Es crea en el cas de que no estigui creada.
    * La carpeta `FMM-Output` (dins de `Output-Data`), la qual tindrà tot el recull de *tracks* de sortida que s'obtindran utilitzant l'algoritme definit a la segona part del processat. Es crea en el cas de que no estigui creada.
    * La carpeta `Cleaned-Output` (dins de `Output-Data`), que contindrà tots els fitxers de *tracks* de sortida després d'aplicar l'algoritme de la tercera part del processat. Es crea en el cas de que no estigui creada.
    * La carpeta `Logs` (dins de `Output-Data`), amb fitxers que contindran el registre de cada execució.
    * El *dataframe* `discard_files` (dins de `Output-Data/Dataframes`), el qual contindrà un registre dels identificadors dels *tracks* descartats i del tipus d'error que s'ha donat per a descartar-los. Aquest es crea buit, o es llegeix.
    * El *dataframe* `output_files` (dins de `Output-Data/Dataframes`), que té la informació amb la qual s'ha dut a terme el processat de *Fast Map Matching* (identificador del *track* i paràmetres per a l'algoritme - `k`, `radius`, i `gps_errror`). Aquest es crea buit, o es llegeix.

4. Utilitzem la funció `create_edges_df()`, que té com entrada les dades d'*Open Street Map*, i l'enllaç a la carpeta `Dataframes` per a crear el *dataframe* `edges` de la següent forma i estructura:

    * Llegim el fitxer *shapefile* que conté els eixos amb `geopandas`. D'aquest n'utilitzem les columnes `u`, `v` i `geometry` (identificadors d'inici i final de l'eix, i geometria).
    * Degut a que alguns eixos són d'anada i tornada; és a dir, tenen la mateixa geometria i identificadors, però al revés. Reordenem aquests identificadors de més petit (`u`) a més gran (`v`), de forma que ens quedem sols amb una fila d'aquelles repetides.
    * Afegim una columna anomenada `id` que conté un identificador numèric propi per a cada eix, la columna `total_tracks` (inicialitzada amb 0), que en un futur modificarem per a tenir el nombre total de rutes que passen per aquell eix, `list_tracks` (inicialitzada amb una llista buida), que anirem omplint amb aquells camins que passen per aquell eix, i `weight`, que serà la columna `total_tracks` normalitzada de 1 a 10.

5. Amb la funció `create_cleaned_output_df()` crearem el *dataframe* `cleaned_out` (dins de `Output-Data/Dataframes`), que anirem omplint a la tercera part del processat amb informació rellevant sobre la ruta (usuari, data, clima, ...). 
6. Finalment, cridarem a `create_waypoints_df()`, que crearà el *dataframe* `waypoints`(dins de `Output-Data/Dataframes`), el qual tindrà la informació sobre aquells punts d'interés on passen les rutes. 

Amb aquests passos haurem creat toes les carpetes i algunes altres dades d'entrada que necessitàvem. Amb tota aquesta informació podrem començar a executar l'algoritme de *Fast Map Matching* definit a la segona part del codi.



