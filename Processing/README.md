# PROCESSING

Aquesta carpeta conté tots els fitxers necessaris per al processat de les dades d'entrada per a poder obtenir tota la informació necessaria per a poder mostrar les visualitzacions desitjades. El codi ha estat executat per les tres zones de les quals es té informació:

* El Canigó (fitxer d'entrada: `canigo.zip`). Zona definida per les fites `[(42.4, 2.2)- (42.6, 2.7)]`. `x` fitxers inicials de *tracks*.
* Matagalls (fitxer d'entrada `matagalls.zip`). Zona definida per les fites `[(41.8, 2.3)- (41.8, 2.5)]`. `x` fitxers inicials de *tracks*.
* Vall Ferrera (fitxer d'entrada `vallferrera.zip`). Zona definida per les fites `[(42.5, 1.2)- (42.8, 1.7)]`. `x` fitxers inicials de *tracks*.

Com podem veure, l'entrada del codi serà un fitxer *zip* que contindrà tots els *tracks* d'entrada per la zona definida. Degut a que cada carpeta conté molts *tracks*, s'ha adaptat el codi per tal de poder anar processant rutes en execucions diferents; és a dir, es va mantenint un registre de les rutes processades per tal de no processar rutes ja tractades en execucions posteriors. 

A continuació, s'explicarà tot el procés realitzat, el qual es troba dividit en tres parts: **PREPROCESSING**, **FMM-PROCESSING** i **POSTPROCESSING**. També es comentarà l'estructura de les dades d'entrada i les de sortida. 

## 1. PREPROCESSING

Aquesta primera part del processat es basa en definir totes les adreces de totes les dades i llegir totes les bases de dades per a poder-les utilitzar en un futur. Podem utilitzar aquest codi per a crear les carpetes i els fitxers en una execució inicial, o llegir totes les dades i *paths* en execucions posteriors. Es pot trobar el codi a [zone_preprocessing.py](zone_preprocessing.py).

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

## 2. FMM-PROCESSING

La segona part del processat de les dades es basa en aplicar l'algoritme de *Fast Map Matching* per a trobar els camins d'*Open Street Map* pels quals ha passat la ruta processada. S'utilitza la llibreria `fmm` ([enllaç amb documentació](https://fmm-wiki.github.io/)), la qual per a cada coordenada d'entrada n'obté una de sortida que correspon a un camí de la xarxa definida (creada amb `osmnx`). L'algoritme que la llibreria utilitza, fa servir tres paràmetres d'entrada:

* El nombre de candidats `k`: aquest correspon al nombre d'opcions inicials de *matching point* que la llibreria troba. Al final, es queda amb aquella amb menys error.
* El radi de cerca `radius`: donat el punt d'entrada, mira en aquest radi tots els candidats especificats (en graus).
* L'error de GPS `gps_error`: correspon a l'error que li permet al GPS (en graus). 

A continuació, explicarem el codi seguit en aquesta segona part del processat, juntament amb els paràmetres d'entrada de `fmm` utilitzants. Hem de tenir en compte que a l'inici d'aquest, s'ha cridat la funció que crida a la part de **PREPROCESSING**, per tant, tindrem tots els dataframes definits. El codi es pot trobar a [zone_fmm.py](zone_fmm.py).

1. Un cop cridada la funció principal de la part de preprocessat, creem la xarxa i el graf de camins que necessitarem com a entrada per a `fmm`. També creem (o llegim) el fitxer `UBODT` (també necessari). Finalment, amb la xarxa, el graf, i el fitxer, podem crear el model que necessitem. 
2. Obtenim una llista amb aquells *tracks* ja processats (o bé es troben al *dataframe* de sortida o al de fitxers descartats). Amb aquesta llista, comprovarem, abans de processar un fitxer, que no hagi estat processat anteriorment. 

*A partir d'aquí, entrarem a un procés de tractament per a cada fitxer que es trobi a la carpeta d'entrada. Com ja hem comentat, el fitxer serà tractat únicament si no ha estat abans.*

3. Utitlizem la funció `extract_information` per a, mitjançant el *path* del fitxer *json* de la ruta, obtenim els *dataframes* de coordenades i de punts d'interés, i també el tipus d'activitat (utilitzat posteriorment). 
4. En el cas de que el tipus d'activitat sigui "*Senderisme*" procedirem amb el processat. En canvi, descartarem el fitxer (**tipus d'error 6**). Això ho fem per homogeneitzar la base de dades de sortida. Ens interessa fer un estudi per a un perfil d'usuaris concret.
5. Utilitzarem la funció `discard_coordinates()` que ens ajudarà a definir si el fitxer d'entrada el processarem, o no. Aquesta funció:

    * Comprova que totes les coordenades es trobin dins de les fites de la zona definida. En el cas contrari, descartarem el fitxer (**tipus d'error 1**).
    * S'assegura que hi ha un mínim de 100 coordenades al *dataframe*. En el cas contrari, descartarem el fitxer (**tipus d'error 2**).
    * Mira que, entre dos coordenades consecutives, no hi hagi una distància superior a 300 metres. Si hi és, descartarem el fitxer (**tipus d'error 3**).
    * Comprova que la distància total del la ruta sigui superior a 1000 metres. En el cas contrari, descartarem el fitxer (**tipus d'error 4**).

6. Ara, si la funció anterior ens permet procedir amb el processat del fitxer, durem a terme el procés de *Fast Map Matching* utilitzant la funció `matching_track()`, la qual utilitza el *dataframe* de coordenades i el model definit, i ens retorna el resultat (en format de `fmm`), els paràmetres utilitzats, i si hi ha hagut algun error.

    * Els paràmetres que utilitzarem per a cridar el *Fast Map Matching* seran, inicialment: `k=2`, `radius=0.001` (aproximadament 100 metres) i `gps_error=0.001` (aproximadament 100 metres). 
    * Transformarem les coordenades del *dataframe* d'entrada en un format adient, i aplicarem la funció `match_wkt()` de la llibreria. 
    * En el cas de trobar algun resultat, el retornarem. En canvi, si no el troba, augmentarem el valor de `k` (fins arribar a `k=4`). Donarem, també, un *timeout* de 60 segons per a poder aplicar la funció.
    * En cas de no haver pogut trobar cap *matching track*, retornarem que el fitxer s'ha de descartar (**tipus d'error 5**).

7. Si hem pogut processar el *track*, en guardarem els punts d'interés al *dataframe* `waypoints` comú per a totes les rutes d'una zona. Utilitzem la llibreria `waypoints_df_cleaning()`, amb la qual netejem el *dataframe* d'entrada quedant-nos només amb informació rellevant.
8. Amb la funció `save_fmm_output()` utilitzem el fitxer resultat del *Fast Map Matching* i els guardem en format *csv* a la carpeta `FMM-Output`. Per a cada punt trobat en guardem les coordenades i també l'eix per on passa (`u` i `v` igual que al *dataframe* d'eixos - també reordenem aquests valors igual que abans).
9. Finalment, guardem les dades als *dataframe* de fitxers de sortida o de fitxers descartats. 

Un cop executat aquest segon procés, haurem omplert la carpeta `FMM-Output` amb fitxers *csv* de tots aquells *tracks* que s'han pogut processar. També hem omplert els *dataframes* `waypoints`, `output_files` i `discard_files`. Ara, amb totes aquestes dades, podrem acabar amb la tercera part del processat. 

## 3. POSTPROCESSING

La tercera i última part del processat de les dades es basa en fet un netejat de les dades extretes a la segona part del procés general. Es basa en dos procediments: el primer agafa cada *track* de sortida i el neteja extreient informació rellevant; i el segon mira un altre cop tots aquests fitxers i va completant el *dataframe* `edges`. És necessari mirar els *tracks* de sortida dos cops. A continuació veurem perquè. Tot aquest procés es troba a [zone_postprocessing.py](zone_postprocessing.py).

Com ja hem comentat, el primer sub-procés neteja cada *track* que anteriorment hem extret:

1. Inicialment, llegim el fitxer *json* inicial i obtenim informació com ara `url`, `user`, `date` (la qual transformem a un format correcte amb `convert_date()`), i `difficulty`. 
2. Utilitzem la funció `clean_coordinates_df()` per a netejear el *dataframe* de coordenades. Aquesta funció té com entrada el *dataframe* de coordenades inicial, el de sortida de `fmm`, i el d'eixos.

    * A partir del *dataframe* inicial, n'agafem les coordenades i l'elevació. Afegim altres columnes com ara `dist` (distància acumulativa a mesura que van passant coordenades - calculem la distància entre dos punts utilitzant `geodesic`), `km` (conté el tram de quilòmetre on es troben aquelles coordenades de la ruta). 
    * Amb la sortida de `fmm` (recordem que tenim coordenades i punts dels eixos), fem un *merge* amb el *dataframe* d'eixos per tal d'obtenir l'indentificador de l'eix per on es passa.
    * A la funció `clean_single_edges()` li passem el *merged dataframe* anterior. Amb aquesta evitem que hi hagi eixos aïllats; és a dir, que es visitin un cop en un veïnat (degut a potser un error de la llibreria). Insertem l'identificador anterior. 
    * Concatenem aquest *dataframe* de sortida amb l'incial, i creem la columna `id` amb un identificador que enumera cada punt de coordenades. Així doncs, tindrem un *dataframe* final amb columnes `id`, `edge_id`, `lon`, `lat`, `elev`, `dist`, `km`, `clean_lon`, i `clean_lat`. 
    * Retornem, juntament amb el *dataframe*, la distància total, l'elevació guanyada i perduda, i les primeres i últimes coordenades (informació que afegirem al *dataframe* comú `cleaned_out`).
    

