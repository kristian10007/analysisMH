PreprocessedData=data/preproceesed_data_Michelle.csv
Indices=data/final_cluser_patient_id.npy
Now=$(shell date '+%Y-%m-%d_%H-%M-%S')
LIB=config.py tools.py revdict.py
Report=data/report.pdf

ALL: out/analysis_Michelle_$(Now).zip


out/analysis_Michelle_%.zip: $(PreprocessedData) $(Indices) $(Report)
	test -d out || mkdir out
	zip $@ data/* data/images/* data/images/*/* data/images/*/*/*


$(PreprocessedData): data/step1.md

$(Indices): data/step3.md

data/%.pdf: data/%.md
	pandoc -f markdown -t latex --toc $< -o $<.tex
	pandoc -f markdown -t pdf --toc $< -o $@

data/report.md: data/step1.md data/step2.md data/step3.md data/step4.md
	cat data/step1.md > data/report.md
	echo '' >> data/report.md
	cat data/step2.md >> data/report.md
	echo '' >> data/report.md
	cat data/step3.md >> data/report.md
	echo '' >> data/report.md
	cat data/step4.md >> data/report.md


report.sh:
	echo '#/bin/bash' > $@
	echo 'echo "=="' >> $@
	echo 'echo "== $$1"' >> $@
	echo 'echo "============================"' >> $@
	chmod +x report.sh

# Step 1
data/step1.md: report.sh Tabelle_mit_Patientennummer_K.xlsx step1*.py $(LIB)
	./report.sh 'Preprocessing data'
	rm -f $@ $(PreprocessedData) data/rev_dict.json
	test -d data || mkdir data
	python3 step1_*.py > data/step1.log 2>&1
	touch $@


# Step 2
data/step2.md: $(PreprocessedData) step2*.py $(LIB)
	./report.sh 'Classification'
	rm -rf $@ data/images/Classification
	python3 step2_*.py > data/step2.log 2>&1
	touch $@


# Step 3
data/step3.md: $(PreprocessedData) step3*.py $(LIB)
	./report.sh 'Stratification'
	rm -rf $@ data/images/Stratification data/*.npy
	python3 step3_*.py > data/step3.log 2>&1
	touch $@


# Step 4
data/step4.md: $(PreprocessedData) $(Indices) step4*.py $(LIB)
	./report.sh 'Hypothesis testing'
	rm -rf $@ data/images/cluster_hypothesis_testing
	python3 step4_*.py > data/step4.log 2>&1
	touch $@


