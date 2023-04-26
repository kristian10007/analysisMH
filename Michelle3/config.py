import numpy as np
import os.path

#
fileName_csv_preprocessed = 'data/preproceesed_data_Michelle.csv'
fileName_final_cluser_indexes = 'data/final_cluser_indexes.npy'
fileName_final_cluser_patient_id = 'data/final_cluser_patient_id.npy'

# column name keys
c_1_NTX = '1. NTX'
c_1_Jahr_post_OP = '1 Jahr post OP'
c_2_Wochen_post_OP = '2 Wochen post OP'
c_3_Monate_post_OP = '3 Monate post OP ymol/l'
c_Abstossungsreaktion = 'Abstoßungsreaktion'
c_Alter_bei_Spende = 'Alter bei Spende'
c_Alter_bei_Tx = 'Alter bei Tx'
c_Atemwegsinfekt = 'Atemwegsinfekt'
c_ATG = 'ATG (Antithymozytenglobulin)'
c_Ausfuhr_bei_Entlassung = 'Ausfuhr bei Entlassung (ml)'
c_Azathioprin = 'Azathioprin'
c_Banff = "Banff"
c_Basiliximab_Simulect = 'Basiliximab (Simulect)'
c_Biospie = 'Biospie ja/nein'
c_Blutgruppe_Empfaenger = 'Blutgruppe Empfänger'
c_Blutgruppe_Spender = 'Blutgruppe  Spender'
c_BMI_Empfaenger = 'BMI Empfänger'
c_BMI_Spender = 'BMI Spender'
c_C_T = 'C --> T'
c_Cell_Myfortic = 'Cell. --> Myfortic'
c_Cellcept_CyA = 'Cellcept+CyA'
c_Cellcept_Myfortic = 'Cellcept+Myfortic'
c_CMV = 'CMV'
c_CMV_Empfaenger = 'CMV Empfänger'
c_CMV_Empfaeger_Spender = 'CMV Empfäger - Spender'
c_CMV_Positiv_Spender = 'CMV Positiv  = 1 (Spender)'
c_CMV_Spender = 'CMV Spender'
c_COPD_Asthma = 'COPD/Asthma'
c_Cross_match = 'Cross match'
c_Cyc_Spiegel_bei_Entlassung = 'Cyc Spiegel bei Entlassung (yg/l)'
c_Cyc_Spiegel_bei_Nachsorge = 'Cyc Spiegel bei Nachsorge'
c_Datum_Explantation = 'Datum Explantation'
c_Dauer_in_Minuten = 'Dauer in Minuten'
c_Diabetes_mellitus = 'Diabetes mellitus'
c_Diabetes_Folgeschaeden = 'Diabetes Folgeschäden '
c_dialysefrei = 'dialysefrei ja/nein'
c_Dialysezeit = 'Dialysezeit (Tage)'
c_Donozentrum = 'Donozentrum'
c_erste_Ausfuhr_am = 'erste Ausfuhr am:'
c_Everolimus = 'Everolimus'
c_Explantation = 'Explantation'
c_Fehlgeburten = 'Fehlgeburten'
c_Geburten = 'Geburten '
c_Geburten_und_co = 'Geburten und co. Ja/nein'
c_Geschlecht_Empfaenger = 'Geschlecht Empfänger'
c_Geschlecht_Spender = 'Geschlecht Spender'
c_GFR_bei_Entlassung = 'GFR bei Entlassung'
c_Grad = "Grad"
c_Groesse_Empfaener = 'Größe Empfänger'
c_Groesse_Spender = 'Größe Spender'
c_Gewicht_Empfaener = 'Gewicht Empfänger'
c_Gewicht_Spender = 'Gewicht Spender'
c_Grund_fuer_TX = 'Grund für TX'
c_Herzerkrankung = 'Herzerkrankung'
c_HLA_A_Mismatch = 'HLA A Mismatch'
c_HLA_B_Mismatch = 'HLA B Mismatch'
c_HLA_DR_Mismatch = 'HLA DR Mismatch'
c_HWI = "HWI"
c_Hypertonie = 'Hypertonie'
c_Immunologische_Erkrankungen = 'Immunologische Erkrankungen'
c_Immunologisches_Grading = 'Immunologisches Grading'
c_Immunadsoprtionstherapie = 'Immunadsoprtionstherapie '
#c_Infektionen = 'Infektionen'
c_Infektionen_insgesamt = 'Infektionen insgesamt'
c_kalte_Ischaemiezeit = 'kalte Ischämiezeit (h)'
c_Katheterinfektion = 'Katheterinfektion'
c_KHK = 'KHK '
c_klinisches_Grading = 'klinisches Grading'
c_Kreatinin_bei_Entlassung = 'Kreatinin bei Entlassung (ymol/l)'
c_Konversion_auf_andere = 'Konversion auf andere'
c_MaFr1FrMa2FrFr3MaMa4 = 'MaFr1FrMa2FrFr3MaMa4'
c_Malignom_vor_OP = 'Malignom vor OP'
c_Malignom_nach_OP = 'Malignom nach OP'
c_Malignome_insgesamt = 'Malignome insgesamt'
c_MDT = 'M-D-T'
#c_Medikamenten_NW = 'Medikamenten-NW'
c_mf_Empfaenger = 'm = 1 ; f = 0 (Empfänger)'
c_mf_Spender = 'm = 1 ; f = 0 (Spender)'
c_MMF = 'MMF'
c_NTx_Anzahl = 'NTx-Anzahl'
c_OP_Tag = 'OP-Tag'
c_pAVK = 'pAVK'
c_Patientennummer = 'Patientennummer'
c_Pilzinfektion = 'Pilzinfektion'
c_po_Tag_dialysefrei = 'p.o. Tag dialysefrei'
c_post_OP_Dialysen = 'post OP Dialysen'
c_post_OP_Dialysen_yn = 'post OP Dialyse ja/nein'
c_Praeformierte_AK = 'Präformierte AK %'
c_Prednisolon = 'Prednisolon'
c_Prograf_Cellcept = 'Prograf+Cellcept'
c_Prograf_Myfortic = 'Prograf+Myfortic'
c_Range_Explantation = 'Range_Explantation'
c_Range_Gestorben = 'Range_Gestorben'
c_Revisions_OP = 'Revisions OP'
c_Rh_Compatibility = 'Rh Compatibility'
c_Rh_Empfaenger = 'Rh Empfänger'
c_Rh_Spender = 'Rh Spender'
c_Rituximab_Immunglobuline = 'Rituximab/ Immunglobuline'
c_Schwangerschaftsabbrueche = 'Schwangerschaftsabbrüche'
c_Sepsis = 'Sepsis'
c_Sirolimus = 'Sirolimus'
c_Sonstige = 'Sonstige'
c_Spenderprogramm = 'Spenderprogramm'
c_T_C = 'T --> C'
c_Tacrolimus = 'Tacrolimus'
c_Tacrolimus_Spiegel_bei_Entlassung = 'Tacrolimus Spiegel bei Entlassung (ng/ml)'
c_Tacrolimus_Spiegel_Nachkontrolle = 'Tacrolimus Spiegel Nachkontrolle'
c_TC_switch = 'TC_switch'
c_Transfusionen_in_der_Vergangenheit = 'Transfusionen in der Vergangenheit'
c_Todesursache = 'Todesursache'
c_Todspende = "Todspende"
c_Urbanstosstherapie = 'Urbanstoßtherapie'
c_Virusinfektionen = 'Virusinfektionen'
c_Wundheilungsstoerung = 'Wundheilungsstörung'
c_WIZ = 'WIZ'
c_ZAHL_Empfaenger = 'ZAHL Empfänger'


# Mappings
map_yes_no = { 'ja': 1, 'ja ': 1, 'nein': 0, 'nein ': 0, '###': np.nan, 'x': np.nan }
map_pos_neg = { 'positiv': 1, 'posititv': 1, 'negativ': 0 }
map_geschlecht = {'männlich': 1, 'weiblich': 0}
map_x_NaN = {'x': np.nan, 'X': np.nan, '###': np.nan, '####': np.nan, '-': np.nan}
map_blutgruppe = {"A ": 4, 'AB':1, 'B': 2, 0:3, 'A':4, 'd':np.nan}

rename_columns = { 'Malignom': c_Malignom_vor_OP
                 , 'Malignome': c_Malignom_nach_OP
                 }

columns_to_remove = [ 'andere'
                    , 'Art der Abstoßungsreaktion'
                    , 'Besonderheiten'
                    , 'Biopsie'
                    , 'BMI Empfänger '
                    , 'Dialyse Anfang'
                    , c_Geschlecht_Empfaenger
                    , c_Geschlecht_Spender
                    , 'Histologische Klassifikation'
                    , 'HLA A = 0'
                    , 'HLA A = 1'
                    , 'HLA A = 2'
                    , 'HLA B = 0'
                    , 'HLA B = 1'
                    , 'HLA B = 2'
                    , 'HLA DR = 0'
                    , 'HLA DR = 1'
                    , 'HLA DR = 2'
                    , 'HLA-Mismatch'
                    , 'Infektionen'
                    , 'Medikamenten-NW'
                    , 'Revisions OP.1'
                    , 'Spender BMI'
                    , 'Uhrzeit OP (Schnitt)'
                    , 'Uhrzeit OP-Ende'
                    , 'ZAHL'
                    ]

data_mapping = { c_Blutgruppe_Empfaenger: map_blutgruppe
               , c_Blutgruppe_Spender: map_blutgruppe
               , c_CMV_Empfaenger: map_pos_neg
               , c_CMV_Spender: map_pos_neg
               , c_Cross_match: map_pos_neg
               , c_erste_Ausfuhr_am: {"sofort": 2, 'verzögert':1, ' verzögert':1, 'keine':0, 'nie':0}
               , c_Rh_Spender: {"D": 1, 'd':0}
               , c_Rh_Empfaenger: {"D": 1, 'd':0, 'B': 2}
               , c_Wundheilungsstoerung: {'nein': 0, 'ja': 1, "x": 1}
               }

data_mapping_bool_x_is_yes = [ c_Atemwegsinfekt
                             , c_ATG
                             , c_Azathioprin
                             , c_Basiliximab_Simulect
                             , 'Belatacept'
                             , c_C_T
                             , c_Cell_Myfortic
                             , 'CELLCEPT'
                             , c_Cellcept_CyA
                             , c_Cellcept_Myfortic
                             , 'Cellcept+Rapamune+Sandimmun'
                             , 'Cellcept+Sanimmun'
                             , 'Ciclosporin'
                             , c_CMV
                             , c_CMV_Empfaeger_Spender
                             , c_COPD_Asthma
                             , 'CyA+ Everolimus'
                             , 'CyA+Myfortic'
                             , 'CyA+Myfenax'
                             , 'CyA+Sandimmun'
                             , 'CyA+Sandimmun+Cellcept'
                             , 'Cyclosporin A'
                             , 'Envarsus (Tacrolimus) + Cellcept'
                             , c_Everolimus
                             , c_Herzerkrankung
                             , c_HWI
                             , c_Immunadsoprtionstherapie
                             , c_Immunologische_Erkrankungen
                             , 'Imuran+CyA'
                             , c_Katheterinfektion
                             , c_KHK
                             , c_Konversion_auf_andere
                             , c_Malignom_vor_OP
                             , c_MDT
                             , c_MMF
                             , c_pAVK
                             , c_Pilzinfektion
                             , c_Prednisolon
                             , 'PROGRAF'
                             , 'Prograf+CyA'
                             , c_Prograf_Cellcept
                             , 'Prograf+Cellcept+Sandimmun'
                             , 'Prograf+Cellcept+Rapamune'
                             , 'Prograf + Everolimus'
                             , 'Prograf +Rapamune'
                             , 'RAPAMUNE (Sirolimus) + Cellcept'
                             , 'Rapamune+Mayfortic'
                             , c_Rituximab_Immunglobuline
                             , c_Prograf_Myfortic
                             , 'Sandimmun+Myfortic'
                             , 'Sandimmun+Rapamune'
                             , c_Sepsis
                             , c_Sirolimus
                             , c_Sonstige
                             , c_T_C
                             , c_Tacrolimus
                             , c_Todspende
                             , c_Urbanstosstherapie
                             , c_Virusinfektionen
                             ]

data_mapping_bool_yes_no = [ c_1_NTX
                           , c_Abstossungsreaktion
                           , c_Biospie
                           , c_Diabetes_mellitus
                           , c_Diabetes_Folgeschaeden
                           , c_dialysefrei
                           , c_Explantation
                           , c_Geburten_und_co
                           , c_Hypertonie
                           , c_post_OP_Dialysen_yn
                           , c_Revisions_OP
                           , c_Transfusionen_in_der_Vergangenheit
                           ]

data_mapping_x_is_nan = [ c_1_Jahr_post_OP
                        , c_2_Wochen_post_OP
                        , c_3_Monate_post_OP
                        , c_Abstossungsreaktion
                        , c_Ausfuhr_bei_Entlassung
                        , c_Cyc_Spiegel_bei_Nachsorge
                        , c_Dialysezeit
                        , c_Fehlgeburten
                        , c_Groesse_Spender
                        , c_Gewicht_Spender
                        , c_Immunologisches_Grading
                        , c_Infektionen_insgesamt
                        , c_kalte_Ischaemiezeit
                        , c_Kreatinin_bei_Entlassung
                        , c_Malignome_insgesamt
                        , c_Praeformierte_AK
                        , c_Schwangerschaftsabbrueche
                        , c_Tacrolimus_Spiegel_Nachkontrolle
                        , c_Todesursache
                        , c_WIZ
                        ]

data_mapping_auto_count = [c_Spenderprogramm, c_Donozentrum]

# ===========================================================================
        
# ===========================================================================

global cont_features 
global nom_features 
global ord_features 


def updateMissingColumns(data):
    global cont_features 
    global nom_features 
    global ord_features

    all_labeled_columns = cont_features + nom_features + ord_features

    for x in all_labeled_columns:
        if x not in data.columns:
            print(f"Column not in dataset: '{x}'")
            if x in cont_features:
                cont_features.remove(x)

            if x in nom_features:
                nom_features.remove(x)

            if x in ord_features:
                ord_features.remove(x)




# feature classes
cont_features = [ c_Alter_bei_Tx
                , c_1_Jahr_post_OP
                , c_2_Wochen_post_OP
                , c_3_Monate_post_OP
                , c_Alter_bei_Spende
                , c_Ausfuhr_bei_Entlassung
                , c_BMI_Empfaenger
                , c_BMI_Spender
                , c_Cyc_Spiegel_bei_Entlassung
                , c_Cyc_Spiegel_bei_Nachsorge
                , c_Dauer_in_Minuten
                , c_Dialysezeit
                , c_Groesse_Empfaener
                , c_Gewicht_Empfaener
                , c_Groesse_Spender
                , c_Gewicht_Spender
                , c_GFR_bei_Entlassung
                #, 'GFR_2Wochen_postOP'
                #, 'GFR_3_Monate_post_OP'
                #, 'GFR_1_Jahr_post_OP'
                , c_Kreatinin_bei_Entlassung
                , c_kalte_Ischaemiezeit
                , c_Praeformierte_AK
                , c_Tacrolimus_Spiegel_bei_Entlassung
                , c_Tacrolimus_Spiegel_Nachkontrolle
                , c_WIZ
                ]

nom_features = [ c_1_NTX
               , c_Banff
               , c_Blutgruppe_Empfaenger
               , c_Blutgruppe_Spender
               , c_CMV_Empfaenger
               , c_CMV_Spender
               , c_Cross_match
               , c_Cell_Myfortic
               , c_Donozentrum
               , c_Grad
               , c_Grund_fuer_TX
               , c_Konversion_auf_andere
               , c_MaFr1FrMa2FrFr3MaMa4
               , c_mf_Empfaenger
               , c_mf_Spender
               , c_post_OP_Dialysen_yn
               , c_Rh_Compatibility
               , c_Spenderprogramm
               , c_TC_switch
               , c_Todesursache
               ]

ord_features = [ c_Abstossungsreaktion
               , c_Atemwegsinfekt
               , c_ATG
               , c_Azathioprin
               , c_Basiliximab_Simulect 
               , c_Biospie
               , c_Cellcept_CyA
               , c_Cellcept_Myfortic
               , c_CMV
               , c_CMV_Positiv_Spender
               , c_COPD_Asthma
               , c_Diabetes_Folgeschaeden
               , c_Diabetes_mellitus
               , c_dialysefrei 
               , c_erste_Ausfuhr_am
               , c_Everolimus
               , c_Explantation 
               , c_Fehlgeburten
               , c_Geburten
               , c_Geburten_und_co
               , c_Herzerkrankung 
               , c_HLA_A_Mismatch
               , c_HLA_B_Mismatch
               , c_HLA_DR_Mismatch
               , c_Hypertonie 
               , c_HWI
               , c_MDT
               , c_Immunologische_Erkrankungen 
               , c_Immunologisches_Grading
               , c_Immunadsoprtionstherapie
               , c_Infektionen_insgesamt
               , c_Katheterinfektion
               , c_KHK
               , c_klinisches_Grading
               , c_Malignom_nach_OP
               , c_Malignom_vor_OP
               , c_Malignome_insgesamt
               , c_MMF
               , c_NTx_Anzahl
               , c_pAVK
               , c_Pilzinfektion
               , c_post_OP_Dialysen 
               , c_po_Tag_dialysefrei
               , c_Prednisolon
               , c_Prograf_Cellcept
               , c_Prograf_Myfortic
               , c_Range_Explantation
               , c_Range_Gestorben
               , c_Revisions_OP 
               , c_Rituximab_Immunglobuline
               , c_Schwangerschaftsabbrueche
               , c_Sepsis
               , c_Sirolimus
               , c_Sonstige
               , c_Todspende
               , c_Transfusionen_in_der_Vergangenheit
               , c_Urbanstosstherapie
               , c_Virusinfektionen
               , c_Wundheilungsstoerung
               , c_ZAHL_Empfaenger
               ]

ignore_features = [c_C_T, c_T_C]

for x in data_mapping_bool_x_is_yes:
    if x not in cont_features and x not in nom_features and x not in ord_features and x not in ignore_features:
        nom_features.append(x)

