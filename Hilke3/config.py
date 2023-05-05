cont_features = [ 'Dialysezeit in Tagen '
    , 'Alter bei Tx (Empfänger)'
    , 'BMI','Präformierte AK in %','OP Dauer in Minuten'
    , 'WIZ','kalte Ischämiezeit (Minuen)'
    , 'Ausfuhr bei Entlassung (in ml)'
    , 'Kreatinin bei Entlassung µmol/l'
    , 'GFR bei Entlassung '
    , 'Kreatinin 2 Wochen post OP µmol/l'
    , 'GFR nach 2 Wochen '
    , 'Kreatinin 3 Monate post OP'
    , 'GFR nach 3 Monaten'
    , 'Kreatinin 1 Jahr post OP (- dead or lost kidney)'
    , 'GFR nach 1 Jahr '
    , 'Tacrolimusspiegel bei Entlassung in ng/ml'
    , 'Tacrolimusspiegel bei Nachkontrolle in ng/ml'
    , 'Ciclosporinspiegel bei Entlassung in µg/l'
    , 'Ciclosporinspiegel bei Nachuntersuchung in µg/l '
    , 'Alter bei Spende (Spender)'
    , 'BMI Spender '
    , 'Range_Dialysis'
    , 'Range_Gestorben'
    ]

nom_features = [ 'RH-Faktor (Empfänger)'
    , 'ICD-10 Code'
    , 'm=1; f=0 (Empfänger)'
    , 'Banff Klassifikation (- means no rejection, other: categories of rejection)'
    , 'Gradeinteilung '
    , 'männlich=1; weiblich=0 (Spender)'
    , 'Blutgruppe Spender'
    , 'RH-Faktor Spender'
    , 'CMV Empänger-Spender'
    , 'Todesursache'
    , 'MaFr1FrMa2FrFr3MaMa4'
    , 'Blutgruppe Empfänger'
    ]

ord_features = [ 'Gestorben '
    , 'Blutgruppen inkompabilität'
    , 'Rhesus Inkompabilität'
    , 'Diabetes mellitus'
    , 'Diabetes Folgeschäden '
    , 'arterielle Hypertonie '
    , 'Folgeschäden arterielle Hypertonie'
    , 'KHK '
    , 'COPD/Asthma'
    , 'Herzerkrankungen '
    , 'chronische Lebererkrankung'
    , 'pAVK'
    , 'Maligne Neoplasie '
    , 'Immunologische Erkr'
    , 'Klinisches Grading'
    , 'CMV positiv 1 ; negativ 0','Tx-Anzahl','1. NTx'
    , 'Transfusionen in der Vergangenheit'
    , 'Schwangerschaften'
    , 'Grading Immunologie'
    , 'HLA A-Missmatches (HLA 1)'
    , 'HLA B-Missmatch (HLA 1)'
    , 'HLA DR-Missmatch (HLA 2)'
    , 'Explantation'
    , 'Revisions OP'
    , 'Biopsie'
    , 'erste Ausfuhr'
    , 'Post Op Dialyse notwendig '
    , 'Post Op Dialyse weiter '
    , 'HWI'
    , 'Atemwegsinfektion'
    , 'CMV'
    , 'Pilzinfektion'
    , 'Viereninfektion'
    , 'Sepsis'
    , 'Magen-Darm '
    , 'Katheterinfektionen'
    , 'Wunddeheilungsstörungen'
    , 'Sonstiges'
    , 'Infektionen Anzahl'
    , 'Malignom Anzahl'
    , 'Rituximab prä OP'
    , 'Immunabsorption prä OP'
    , 'Urbanosonstoßtherapie'
    , 'ATG'
    , 'MMF'
    , 'Prednisolon'
    , 'Basiliximab'
    , 'Sirolimus'
    , 'Azathioprin'
    , 'Konversion auf Tacrolimus '
    , 'Konversion auf CyA '
    , 'Prograf + CellCept'
    , 'Prograf + Myofortic'
    , 'CellCept + CyA'
    , 'CMV Spender  Positiv = 1; Negativ = 0'
    , 'Todspende = x'
    ]
    

mapYesNo = { 'ja': 1, 'nein': 0 }
mapPosNeg = { "Positive": 1, 'Negative':0 }
mapBanff = {'No rejection': 0,'rejection-4': 4,'rejection-3': 3,'rejection-2': 2,'rejection-other': -1.0}
mapIcd10Code = {'Q61': 9.0,'N03': 8.0,'I12': 7.0, 'N26': 6.0, 'N02': 5.0, 'N04.1': 4.0, 'E11.2': 3.0, 'N04.3': 2.0, 'others':1 }
    
rev_dict = { 'Blutgruppe Empfänger': { "A ": 4, 'AB': 1, 'B': 2, 'O': 3 }
           , 'RH-Faktor (Empfänger)': mapPosNeg
           , 'Rhesus Inkompabilität': mapYesNo
           , 'ICD-10 Code': mapIcd10Code
           , 'Diabetes mellitus': mapYesNo
           , 'Diabetes Folgeschäden ': mapYesNo
           , 'arterielle Hypertonie ': mapYesNo
           , 'Folgeschäden arterielle Hypertonie': mapYesNo
           , 'KHK ': mapYesNo
           , 'COPD/Asthma': mapYesNo
           , 'Herzerkrankungen ': mapYesNo
           , 'chronische Lebererkrankung': mapYesNo
           , 'pAVK': mapYesNo
           , 'Maligne Neoplasie ': mapPosNeg
           , 'CMV positiv 1 ; negativ 0': mapPosNeg
           , '1. NTx': mapYesNo
           , 'Transfusionen in der Vergangenheit': mapYesNo
           , 'Schwangerschaften': mapYesNo
           , 'Explantation': mapYesNo
           , 'Revisions OP': mapYesNo
           , 'Banff Klassifikation (- means no rejection, other: categories of rejection)': mapBanff
           , 'erste Ausfuhr': {"sofort": 2, 'verzögert':1, ' verzögert':1, 'keine':0}
           , 'Post Op Dialyse notwendig ': mapYesNo
           , 'Post Op Dialyse weiter ': mapYesNo
           , 'HWI': mapYesNo
           , 'Atemwegsinfektion': mapYesNo
           , 'CMV': mapYesNo
           , 'Pilzinfektion': mapYesNo
           , 'Viereninfektion': mapYesNo
           , 'Sepsis': mapYesNo
           , 'Magen-Darm ': mapYesNo
           , 'Wunddeheilungsstörungen': mapYesNo
           , 'Sonstiges': mapYesNo
           , 'Immunabsorption prä OP': mapYesNo
           , 'Urbanosonstoßtherapie': mapYesNo
           , 'ATG': mapYesNo
           , 'MMF': mapYesNo
           , 'Prednisolon': mapYesNo
           , 'Basiliximab': mapYesNo
           , 'Prograf + CellCept': mapYesNo
           , 'Prograf + Myofortic': mapYesNo
           , 'CellCept + CyA': mapYesNo
           , 'männlich=1; weiblich=0 (Spender)': {"M": 1, 'W':0}
           , 'Blutgruppe Spender': {"A": 4, 'AB':1, 'B': 2, '0':3}
           , 'RH-Faktor Spender': mapPosNeg
           , 'CMV Spender  Positiv = 1; Negativ = 0': mapPosNeg
           , 'CMV Empänger-Spender':{"pos auf pos": 1, 'pos auf neg':2, 'neg auf pos': 3, 'neg auf neg':4}
           , 'Todspende = x': {"tot": 1, 'lebendig':0}
           , 'MaFr1FrMa2FrFr3MaMa4': {"Ma->Fr": 1, 'Fr->Ma':2, 'Fr->Fr':3, 'Ma->Ma':4}
           }