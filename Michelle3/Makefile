PreprocessedData=data/preproceesed_data_Michelle.csv
Indices=data/final_cluser_patient_id.npy

analysis.zip: $(PreprocessedData) temp/step2.ok $(Indices) temp/step4.ok 
	zip analysis.zip data/* images/* images/*/* images/*/*/*



# Step 1
$(PreprocessedData): Tabelle_mit_Patientennummer_K.xlsx
	python3 step1_pre-processing_michelle.py


# Step 2
temp/step2.ok: $(PreprocessedData)
	python3 step2_patient_Classification_michelle
	test -d temp || mkdir temp
	touch $@


# Step 3
$(Indices): $(PreprocessedData)
	python3 step3_patient_stratification_michelle.py


# Step 4
temp/step4.ok: $(PreprocessedData) $(Indices)
	python3 step4_patient_clusters_hypothesis_testing_michelle.py
	test -d temp || mkdir temp
	touch $@


