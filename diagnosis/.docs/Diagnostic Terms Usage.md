# Diagnostic Terms Usage

## Counting terms across DBs
7 databases were used in this study, namely ['avfad', 'meei', 'svd', 'torgo', 'uaspeech', 'uncommon_voice', 'voiced'].
These DBs in total had 306 diagnostic terms.
We duplicated terms based on typographical variations (e.g. cyst vs cysts, reinke's edema vs reinke edema), translations (e.g. leukoplakia vs leukoplakie), and synonymous terms (e.g. not sure vs unknown, functional voice disorder vs functional dysphonia). After deduplication wer have 273 diagnostic terms across the 7 databases. The different DBs had the following number of diagnostic terms: {'meei': 168, 'svd': 71, 'avfad': 27, 'voiced': 24, 'uaspeech': 4, 'torgo': 2, 'uncommon_voice': 2}.

## Categorisation of DB terms under different classification systems
### Benba_2017
```
	pathological(2):
		neurological_disorder(0):
			cervical_dystonia(0)
			dystonia(0)
			essential_tremors(1)
			functional_neurological_disorder(0)
			generalized_paroxysmal_dystonia(0)
			somatization(0)
		multiple_system_atrophy(0)
		parkinsons(1)
	normal(7)
	unclassified(1)
```

In total 12 terms from DBs were automatically classified, while 261 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(2):**
	- **morbus parkinson:** parkinsons
	- **normal:** normal
- **meei(10):**
	- **essential tremor:** essential_tremors
	- **gesangsstimme:** normal
	- **mixed:** unclassified
	- **morbus parkinson:** parkinsons
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **pathological voice- diagnosis n/a:** pathological
- **svd(3):**
	- **gesangsstimme:** normal
	- **healthy:** normal
	- **morbus parkinson:** parkinsons
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(1):**
	- **healthy:** normal

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(25):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', "hemmoragic reinke's edema", 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'nodular swelling', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'stimmlippenpolyp', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'unilateral recurrent laryngeal nerve (rln) paralysis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(158):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cyst', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', "hemmoragic reinke's edema", 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'nodular swelling', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'stimmlippenpolyp', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(68):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'carcinoma in situ', 'chondrom', 'chordektomie', 'cyst', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'epiglottiskarzinom', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'granulom', "hemmoragic reinke's edema", 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngitis', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'stimmlippenkarzinom', 'stimmlippenpolyp', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(23):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold paralysis)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'hypokinetic dysphonia (vocal fold paralysis)', 'reflux laryngitis']

### Compton_2022
```
	pathological(2):
		inflammatory(0):
			chronic_laryngitis(2)
			reinkes_edema(1)
		mass_lesions(0):
			cancer(0):
				carcinoma_in_situ(1)
				epiglottic_carcinoma(1)
				vocal_fold_carcinoma(1)
			cysts(1)
			granuloma(1)
			nodules(1)
			papilloma(1)
			polyp(1)
		neuro_muscular(0):
			adductor_spasmodic_dysphonia(1)
			benign_essential_tremor(1)
			bilateral_paralysis(1)
			idiopathic_vocal_cord_bowing(0)
			muscle_tension_dysphonia(0)
			parkinsons(1)
			presbylarynges(0)
			unilateral_paralysis(1)
	normal(7)
	unclassified(1)
```

In total 26 terms from DBs were automatically classified, while 247 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(7):**
	- **bilateral recurrent laryngeal nerve (rln) paralysis-peripheral:** bilateral_paralysis
	- **hemmoragic reinke's edema:** reinkes_edema
	- **morbus parkinson:** parkinsons
	- **nodular swelling:** nodules
	- **normal:** normal
	- **stimmlippenpolyp:** polyp
	- **unilateral recurrent laryngeal nerve (rln) paralysis:** unilateral_paralysis
- **meei(17):**
	- **adductor spasmodic dysphonia:** adductor_spasmodic_dysphonia
	- **chronic laryngitis:** chronic_laryngitis
	- **cyst:** cysts
	- **essential tremor:** benign_essential_tremor
	- **gesangsstimme:** normal
	- **hemmoragic reinke's edema:** reinkes_edema
	- **mixed:** unclassified
	- **morbus parkinson:** parkinsons
	- **nodular swelling:** nodules
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **papillom:** papilloma
	- **pathological voice- diagnosis n/a:** pathological
	- **stimmlippenpolyp:** polyp
- **svd(12):**
	- **carcinoma in situ:** carcinoma_in_situ
	- **cyst:** cysts
	- **epiglottiskarzinom:** epiglottic_carcinoma
	- **gesangsstimme:** normal
	- **granulom:** granuloma
	- **healthy:** normal
	- **hemmoragic reinke's edema:** reinkes_edema
	- **laryngitis:** chronic_laryngitis
	- **morbus parkinson:** parkinsons
	- **papillom:** papilloma
	- **stimmlippenkarzinom:** vocal_fold_carcinoma
	- **stimmlippenpolyp:** polyp
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(1):**
	- **healthy:** normal

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(20):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(151):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'pachydermia', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(59):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'chondrom', 'chordektomie', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(23):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold paralysis)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'hypokinetic dysphonia (vocal fold paralysis)', 'reflux laryngitis']

### Cordeiro_2015
```
	pathological(2):
		physiological(0):
			vocal_fold_edema(1)
			vocal_fold_nodules(1)
		neuro_muscular(0):
			vocal_fold_unilateral_paralysis(1)
	normal(7)
	unclassified(1)
```

In total 13 terms from DBs were automatically classified, while 260 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(3):**
	- **nodular swelling:** vocal_fold_nodules
	- **normal:** normal
	- **unilateral recurrent laryngeal nerve (rln) paralysis:** vocal_fold_unilateral_paralysis
- **meei(10):**
	- **gesangsstimme:** normal
	- **mixed:** unclassified
	- **nodular swelling:** vocal_fold_nodules
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **pathological voice- diagnosis n/a:** pathological
	- **vocal fold edema:** vocal_fold_edema
- **svd(2):**
	- **gesangsstimme:** normal
	- **healthy:** normal
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(1):**
	- **healthy:** normal

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(24):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', "hemmoragic reinke's edema", 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'morbus parkinson', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'stimmlippenpolyp', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(158):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cyst', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'essential tremor', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', "hemmoragic reinke's edema", 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'morbus parkinson', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'stimmlippenpolyp', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(69):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'carcinoma in situ', 'chondrom', 'chordektomie', 'cyst', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'epiglottiskarzinom', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'granulom', "hemmoragic reinke's edema", 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngitis', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'morbus parkinson', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'stimmlippenkarzinom', 'stimmlippenpolyp', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(23):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold paralysis)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'hypokinetic dysphonia (vocal fold paralysis)', 'reflux laryngitis']

### daSilvaMoura_2024
```
	pathological(2):
		functional(0):
			closed_rhinophonia(1)
			dysphonia(1)
			functional_dysphonia(1)
			hyperfunctional_dysphonia(1)
			hypofunctional_dysphonia(1)
			hypotonic_dysphonia(1)
			mixed_rhinophonia(1)
			mutational_falsetto(0)
			open_rhinophonia(1)
			psychogenic_dysphonia(1)
			psychogenic_microphony(1)
		organic(0):
			amyotrophic_lateral_sclerosis(1)
			bulbar_paralysis(1)
			carcinoma_in_situ(1)
			central_laryngeal_movement_disorder(1)
			chondroma(1)
			contact_pachydermia(1)
			cordectomy(1)
			diffuse_idiopathic_skeletal_hyperostosis(0)
			down_syndrome(1)
			epiglottis_carcinoma(1)
			gastroesophageal_reflux_disease(1)
			hypopharyngeal_tumor(1)
			intubation_damage(1)
			intubation_granuloma(1)
			laryngeal_tumor(1)
			laryngitis(1)
			laryngocele(1)
			median_cervical_cyst(1)
			monochorditis(1)
			nasopharyngeal_tumor(1)
			papilloma(1)
			parkinsons_syndrome(1)
			recurrent_paralysis(1)
			spasmodic_dysphonia(1)
			superior_laryngeal_nerve_injury(1)
			superior_laryngeal_nerve_neuralgia(1)
			synechia(1)
			vallecular_cyst(1)
			vocal_fold_carcinoma(1)
		organofunctional(0):
			cyst(1)
			granuloma(1)
			leukoplakia(1)
			phonation_nodule(1)
			reinkes_edema(1)
			vocal_fold_polyp(1)
	normal(7)
	unclassified(1)
```

In total 54 terms from DBs were automatically classified, while 219 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(5):**
	- **amyotrophe lateralsklerose:** amyotrophic_lateral_sclerosis
	- **hemmoragic reinke's edema:** reinkes_edema
	- **morbus parkinson:** parkinsons_syndrome
	- **normal:** normal
	- **stimmlippenpolyp:** vocal_fold_polyp
- **meei(17):**
	- **chordektomie:** cordectomy
	- **cyst:** cyst
	- **gastric reflux:** gastroesophageal_reflux_disease
	- **gesangsstimme:** normal
	- **hemmoragic reinke's edema:** reinkes_edema
	- **intubation trauma:** intubation_damage
	- **laryngocele:** laryngocele
	- **mixed:** unclassified
	- **morbus parkinson:** parkinsons_syndrome
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **papillom:** papilloma
	- **pathological voice- diagnosis n/a:** pathological
	- **stimmlippenpolyp:** vocal_fold_polyp
- **svd(46):**
	- **amyotrophe lateralsklerose:** amyotrophic_lateral_sclerosis
	- **bulbärparalyse:** bulbar_paralysis
	- **carcinoma in situ:** carcinoma_in_situ
	- **chondrom:** chondroma
	- **chordektomie:** cordectomy
	- **cyst:** cyst
	- **dysphonie:** dysphonia
	- **epiglottiskarzinom:** epiglottis_carcinoma
	- **funktionelle dysphonie:** functional_dysphonia
	- **gastric reflux:** gastroesophageal_reflux_disease
	- **gesangsstimme:** normal
	- **granulom:** granuloma
	- **healthy:** normal
	- **hemmoragic reinke's edema:** reinkes_edema
	- **hyperfunktionelle dysphonie:** hyperfunctional_dysphonia
	- **hypofunktionelle dysphonie:** hypofunctional_dysphonia
	- **hypopharynxtumor:** hypopharyngeal_tumor
	- **hypotone dysphonie:** hypotonic_dysphonia
	- **intubation trauma:** intubation_damage
	- **intubationsgranulom:** intubation_granuloma
	- **kehlkopftumor:** laryngeal_tumor
	- **kontaktpachydermie:** contact_pachydermia
	- **laryngitis:** laryngitis
	- **laryngocele:** laryngocele
	- **leukoplakie:** leukoplakia
	- **mediale halscyste:** median_cervical_cyst
	- **mesopharynxtumor:** nasopharyngeal_tumor
	- **monochorditis:** monochorditis
	- **morbus down:** down_syndrome
	- **morbus parkinson:** parkinsons_syndrome
	- **n. laryngeus superior läsion:** superior_laryngeal_nerve_injury
	- **n. laryngeus superior neuralgie:** superior_laryngeal_nerve_neuralgia
	- **papillom:** papilloma
	- **phonationsknötchen:** phonation_nodule
	- **psychogene dysphonie:** psychogenic_dysphonia
	- **psychogene mikrophonie:** psychogenic_microphony
	- **rekurrensparese:** recurrent_paralysis
	- **rhinophonie aperta:** open_rhinophonia
	- **rhinophonie clausa:** closed_rhinophonia
	- **rhinophonie mixta:** mixed_rhinophonia
	- **spasmodische dysphonie:** spasmodic_dysphonia
	- **stimmlippenkarzinom:** vocal_fold_carcinoma
	- **stimmlippenpolyp:** vocal_fold_polyp
	- **synechie:** synechia
	- **valleculacyste:** vallecular_cyst
	- **zentral-laryngale bewegungsstörung:** central_laryngeal_movement_disorder
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(1):**
	- **healthy:** normal

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(22):** ['acute laryngitis', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'nodular swelling', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'unilateral recurrent laryngeal nerve (rln) paralysis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(151):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'essential tremor', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'nodular swelling', 'pachydermia', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(25):** ['aryluxation', 'balbuties', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'fibrom', 'frontolaterale teilresektion', 'hyperasthenie', 'internusschwäche', 'juvenile dysphonie', 'mutatio', 'mutationsfistelstimme', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'phonasthenie', 'poltersyndrom', 'psychogene aphonie', 'sigmatismus', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'velopharyngoplastik', 'vox senilis']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(23):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold paralysis)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'hypokinetic dysphonia (vocal fold paralysis)', 'reflux laryngitis']

### deMoraesLimaMarinus_2013
```
	pathological(2):
		others(1):
			cysts(1)
			nodules(1)
			vocal_fold_paralysis(1):
				hyperkinetic_dysphonia_vocal_fold_paralysis(1)
				hypokinetic_dysphonia_vocal_fold_paralysis(1)
		edema(1)
	normal(7)
	unclassified(1)
```

In total 16 terms from DBs were automatically classified, while 257 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(2):**
	- **nodular swelling:** nodules
	- **normal:** normal
- **meei(12):**
	- **cyst:** cysts
	- **generalized edema of larynx:** edema
	- **gesangsstimme:** normal
	- **mixed:** unclassified
	- **nodular swelling:** nodules
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **paralysis:** vocal_fold_paralysis
	- **pathological voice- diagnosis n/a:** pathological
- **svd(3):**
	- **cyst:** cysts
	- **gesangsstimme:** normal
	- **healthy:** normal
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(3):**
	- **healthy:** normal
	- **hyperkinetic dysphonia (vocal fold paralysis):** hyperkinetic_dysphonia_vocal_fold_paralysis
	- **hypokinetic dysphonia (vocal fold paralysis):** hypokinetic_dysphonia_vocal_fold_paralysis

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(25):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', "hemmoragic reinke's edema", 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'morbus parkinson', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'stimmlippenpolyp', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'unilateral recurrent laryngeal nerve (rln) paralysis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(156):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'essential tremor', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', "hemmoragic reinke's edema", 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'morbus parkinson', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'stimmlippenpolyp', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(68):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'carcinoma in situ', 'chondrom', 'chordektomie', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'epiglottiskarzinom', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'granulom', "hemmoragic reinke's edema", 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngitis', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'morbus parkinson', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'stimmlippenkarzinom', 'stimmlippenpolyp', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(21):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'reflux laryngitis']

### FEMH_2018
```
	pathological(2):
		phonotrauma(0):
			cysts(1)
			nodules(1)
			polyps(1)
		laryngeal_neoplasm(0)
		unilateral_vocal_paralysis(1)
	normal(7)
	unclassified(1)
```

In total 14 terms from DBs were automatically classified, while 259 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(4):**
	- **nodular swelling:** nodules
	- **normal:** normal
	- **stimmlippenpolyp:** polyps
	- **unilateral recurrent laryngeal nerve (rln) paralysis:** unilateral_vocal_paralysis
- **meei(11):**
	- **cyst:** cysts
	- **gesangsstimme:** normal
	- **mixed:** unclassified
	- **nodular swelling:** nodules
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **pathological voice- diagnosis n/a:** pathological
	- **stimmlippenpolyp:** polyps
- **svd(4):**
	- **cyst:** cysts
	- **gesangsstimme:** normal
	- **healthy:** normal
	- **stimmlippenpolyp:** polyps
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(1):**
	- **healthy:** normal

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(23):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', "hemmoragic reinke's edema", 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'morbus parkinson', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(157):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'essential tremor', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', "hemmoragic reinke's edema", 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'morbus parkinson', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(67):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'carcinoma in situ', 'chondrom', 'chordektomie', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'epiglottiskarzinom', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'granulom', "hemmoragic reinke's edema", 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngitis', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'morbus parkinson', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'stimmlippenkarzinom', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(23):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold paralysis)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'hypokinetic dysphonia (vocal fold paralysis)', 'reflux laryngitis']

### Firdos_2016
```
	pathological(2):
		vocal_fold_pathology(0):
			cysts(1)
			inflammation(1)
			nodules(1)
			polyps(1)
			swelling(0)
			vocal_cord_injury(0)
			vocal_cord_paralysis(0):
				hyperkinetic_dysphonia_vocal_fold_paralysis(1)
				hypokinetic_dysphonia_vocal_fold_paralysis(1)
		neurological(0):
			brain_tumors(0)
			parkinsons_trauma(1)
	normal(7)
	unclassified(1)
```

In total 17 terms from DBs were automatically classified, while 256 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(4):**
	- **morbus parkinson:** parkinsons_trauma
	- **nodular swelling:** nodules
	- **normal:** normal
	- **stimmlippenpolyp:** polyps
- **meei(13):**
	- **cyst:** cysts
	- **gesangsstimme:** normal
	- **inflammatory disease:** inflammation
	- **mixed:** unclassified
	- **morbus parkinson:** parkinsons_trauma
	- **nodular swelling:** nodules
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **pathological voice- diagnosis n/a:** pathological
	- **stimmlippenpolyp:** polyps
- **svd(5):**
	- **cyst:** cysts
	- **gesangsstimme:** normal
	- **healthy:** normal
	- **morbus parkinson:** parkinsons_trauma
	- **stimmlippenpolyp:** polyps
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(3):**
	- **healthy:** normal
	- **hyperkinetic dysphonia (vocal fold paralysis):** hyperkinetic_dysphonia_vocal_fold_paralysis
	- **hypokinetic dysphonia (vocal fold paralysis):** hypokinetic_dysphonia_vocal_fold_paralysis

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(23):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', "hemmoragic reinke's edema", 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'unilateral recurrent laryngeal nerve (rln) paralysis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(155):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'essential tremor', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', "hemmoragic reinke's edema", 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(66):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'carcinoma in situ', 'chondrom', 'chordektomie', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'epiglottiskarzinom', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'granulom', "hemmoragic reinke's edema", 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngitis', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'stimmlippenkarzinom', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(21):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'reflux laryngitis']

### Kim_2024
```
	pathological(2):
		benign_mucosal_disease(0):
			granuloma(1)
			nodules(1)
			polyps(1)
			reinkes_edema(1)
		laryngeal_cancer(0):
			carcinoma_in_situ(1)
			epiglottic_carcinoma(1)
			vocal_fold_carcinoma(1)
		vocal_fold_paralysis(0):
			hyperkinetic_dysphonia_vocal_fold_paralysis(1)
			hypokinetic_dysphonia_vocal_fold_paralysis(1)
	normal(7)
	unclassified(1)
```

In total 19 terms from DBs were automatically classified, while 254 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(4):**
	- **hemmoragic reinke's edema:** reinkes_edema
	- **nodular swelling:** nodules
	- **normal:** normal
	- **stimmlippenpolyp:** polyps
- **meei(11):**
	- **gesangsstimme:** normal
	- **hemmoragic reinke's edema:** reinkes_edema
	- **mixed:** unclassified
	- **nodular swelling:** nodules
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **pathological voice- diagnosis n/a:** pathological
	- **stimmlippenpolyp:** polyps
- **svd(8):**
	- **carcinoma in situ:** carcinoma_in_situ
	- **epiglottiskarzinom:** epiglottic_carcinoma
	- **gesangsstimme:** normal
	- **granulom:** granuloma
	- **healthy:** normal
	- **hemmoragic reinke's edema:** reinkes_edema
	- **stimmlippenkarzinom:** vocal_fold_carcinoma
	- **stimmlippenpolyp:** polyps
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(3):**
	- **healthy:** normal
	- **hyperkinetic dysphonia (vocal fold paralysis):** hyperkinetic_dysphonia_vocal_fold_paralysis
	- **hypokinetic dysphonia (vocal fold paralysis):** hypokinetic_dysphonia_vocal_fold_paralysis

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(23):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'morbus parkinson', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'unilateral recurrent laryngeal nerve (rln) paralysis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(157):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cyst', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'essential tremor', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'morbus parkinson', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(63):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'chondrom', 'chordektomie', 'cyst', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngitis', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'morbus parkinson', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(21):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'reflux laryngitis']

### Sztaho_2018
```
	pathological(2):
		morphological_alteration(0):
			chronical_inflammation_of_larynx(0)
			closure_insufficiency(0)
			cysts(1)
			gastroesophageal_reflux_disease(1)
			laryngeal_paralysis(0):
				bilateral_recurrent_laryngeal_nerve_(rln)_paralysis_peripheral(1)
				hyperkinetic_dysphonia_vocal_fold_paralysis(1)
				hypokinetic_dysphonia_vocal_fold_paralysis(1)
				unilateral_recurrent_laryngeal_nerve_(rln)_paralysis(1)
			laryngitis(1)
			tumors_at_different_points_of_vocal_tract(0)
			tractional_stenosis(0)
			vocal_node(1)
		depression(0)
		functional_dysphonia(1)
		parkinsons(1)
		recurrent_paresis(1):
			unilateral_or_bilateral_recurrent_laryngeal_nerve_(rln)_paresis(1)
	normal(7)
	unclassified(1)
```

In total 22 terms from DBs were automatically classified, while 251 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(6):**
	- **bilateral recurrent laryngeal nerve (rln) paralysis-peripheral:** bilateral_recurrent_laryngeal_nerve_(rln)_paralysis_peripheral
	- **morbus parkinson:** parkinsons
	- **nodular swelling:** vocal_node
	- **normal:** normal
	- **unilateral or bilateral recurrent laryngeal nerve (rln) paresis:** unilateral_or_bilateral_recurrent_laryngeal_nerve_(rln)_paresis
	- **unilateral recurrent laryngeal nerve (rln) paralysis:** unilateral_recurrent_laryngeal_nerve_(rln)_paralysis
- **meei(13):**
	- **cyst:** cysts
	- **gastric reflux:** gastroesophageal_reflux_disease
	- **gesangsstimme:** normal
	- **mixed:** unclassified
	- **morbus parkinson:** parkinsons
	- **nodular swelling:** vocal_node
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **paresis:** recurrent_paresis
	- **pathological voice- diagnosis n/a:** pathological
- **svd(7):**
	- **cyst:** cysts
	- **funktionelle dysphonie:** functional_dysphonia
	- **gastric reflux:** gastroesophageal_reflux_disease
	- **gesangsstimme:** normal
	- **healthy:** normal
	- **laryngitis:** laryngitis
	- **morbus parkinson:** parkinsons
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(3):**
	- **healthy:** normal
	- **hyperkinetic dysphonia (vocal fold paralysis):** hyperkinetic_dysphonia_vocal_fold_paralysis
	- **hypokinetic dysphonia (vocal fold paralysis):** hypokinetic_dysphonia_vocal_fold_paralysis

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(21):** ['acute laryngitis', 'amyotrophe lateralsklerose', "hemmoragic reinke's edema", 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'stimmlippenpolyp', 'sulcus vocalis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(155):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'essential tremor', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', "hemmoragic reinke's edema", 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paralysis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'stimmlippenpolyp', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(64):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'carcinoma in situ', 'chondrom', 'chordektomie', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'epiglottiskarzinom', 'fibrom', 'frontolaterale teilresektion', 'granulom', "hemmoragic reinke's edema", 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'stimmlippenkarzinom', 'stimmlippenpolyp', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(21):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'reflux laryngitis']

### Tsui_2018
```
	pathological(2):
		benign_structural_lesions(0):
			cysts(1)
			nodules(1)
			polyps(1)
		neurogenic_dysfunction(0):
			unilateral_vocal_palsy(0)
		neoplasm(0)
	normal(7)
	unclassified(1)
```

In total 13 terms from DBs were automatically classified, while 260 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(3):**
	- **nodular swelling:** nodules
	- **normal:** normal
	- **stimmlippenpolyp:** polyps
- **meei(11):**
	- **cyst:** cysts
	- **gesangsstimme:** normal
	- **mixed:** unclassified
	- **nodular swelling:** nodules
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **pathological voice- diagnosis n/a:** pathological
	- **stimmlippenpolyp:** polyps
- **svd(4):**
	- **cyst:** cysts
	- **gesangsstimme:** normal
	- **healthy:** normal
	- **stimmlippenpolyp:** polyps
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(1):**
	- **healthy:** normal

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(24):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', "hemmoragic reinke's edema", 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'morbus parkinson', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'unilateral recurrent laryngeal nerve (rln) paralysis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(157):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'essential tremor', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', "hemmoragic reinke's edema", 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'morbus parkinson', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(67):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'carcinoma in situ', 'chondrom', 'chordektomie', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'epiglottiskarzinom', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'granulom', "hemmoragic reinke's edema", 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngitis', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'morbus parkinson', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'stimmlippenkarzinom', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(23):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold paralysis)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'hypokinetic dysphonia (vocal fold paralysis)', 'reflux laryngitis']

### USVAC_2025
```
	normal(3):
		without_dysarthria(1)
	pathological(2):
		functional(0):
			functional_dysphonia(1):
				atypical_paradoxical_vocal_fold_movement_of_unknown_etiology(1)
				functional_neurological_disorder(0)
				hypokinetic_dysphonia_conversion_dysphonia(1)
				psychogenic_microphony(1)
				conversion_dysphonia(1)
				episodic_functional_dysphonia(1)
				psychogenic_aphonia(1)
				psychogenic_dysphonia(1)
			functional_puberphonia(0):
				puberphonia(1)
		muscle_tension(1):
			hyperkinetic_dysphonia_nodule(1)
			hyperkinetic_dysphonia_polyps(1)
			hyperkinetic_dysphonia_vocal_fold_nodules(1)
			hyperkinetic_dysphonia(1)
			ventricular_dysphonia(1)
			ventricular_phonation(1)
			hyperfunctional_dysphonia(1)
			hyperfunction(1)
			muscle_tension_adaptive(1):
				hyperkinetic_dysphonia_adduction_deficit(1)
				hyperkinetic_dysphonia_rigid_vocal_fold(1)
		non_laryngeal(0):
			metabolic(0)
			psychiatric(0)
			systemic(0)
		organic(0):
			organic_inflammatory(0):
				organic_inflammatory_infective(0):
					monochorditis(1)
					laryngitis(1)
					acute_laryngitis(1)
					laryngeal_tuberculosis(1)
				organic_inflammatory_non_infective(0):
					post_radiation_difuse_edema_of_entire_larynx(1)
					inflammatory_disease(1)
					non_intubation_related_vocal_fold_granuloma(1)
					post_intubation_submucosal_edema_mild(1)
					contact_granuloma(1)
					teflon_granuloma(1)
					chronic_laryngitis(1)
					inflamed_arytenoid(1)
					inflammation(1)
					cricoarytenoid_arthritis(1)
					reflux_laryngitis(1)
			organic_neuro_muscular(0):
				organic_neuro_muscular_central_nervous_disorder(0):
					hyperkinetic_dysphonia_vocal_fold_paralysis(1)
					hypokinetic_dysphonia_vocal_fold_paralysis(1)
					bulbar_paralysis(1)
					central_laryngeal_movement_disorder(1)
					post_cva_laryngeal_discoordination(1)
					amyotrophic_lateral_sclerosis(1)
					multiple_sclerosis(1)
				organic_neuro_muscular_movement_disorder(0):
					hypokinetic_dysphonia_adduction_deficit(1)
					hypofunctional_dysphonia(1)
					hypokinetic_dysphonia(1)
					hypokinetic_dysphonia_spasmodic_dysphonia(1)
					parkinson_disease(1)
					spasmodic_dysphonia(1)
					abductor_spasmodic_dysphonia(1)
					adductor_spasmodic_dysphonia(1)
					essential_tremor(1)
					mixed_adductor_abductor_spasmodic_dysphonia(1)
					vocal_tremor(1)
				organic_neuro_muscular_peripheral_nervous_disorder(0):
					laryngeal_sensory_dysfunction(0)
					muscular_dystrophy(1)
					recurrent_paralysis(1)
					paralysis(1)
					superior_laryngeal_nerve_lesion(1)
					superior_laryngeal_nerve_neuralgia(1)
					bilateral_recurrent_laryngeal_nerve_rln_paralysis_peripheral(1)
					paresis(1)
					unilateral_or_bilateral_recurrent_laryngeal_nerve_rln_paresis(1)
					unilateral_recurrent_laryngeal_nerve_rln_paralysis(1)
			organic_structural(0):
				organic_structural_congenital(0):
					bowing(1)
					vocal_fold_atrophic(1)
					hypokinetic_dysphonia_presbiphonia(1)
					presbyphonia(1)
				organic_structural_epithelial_propria(0):
					atrophic_laryngitis(1)
					hyperkinetic_dysphonia_reinkes_edema(1)
					phonation_nodules(1)
					post_radiation_fibrosis(1)
					vocal_fold_edema(1)
					keratotic_reaction_to_polyp(1)
					micro_cyst(1)
					pre_nodular_swellings(1)
					vocal_fold_lesion(1)
					multi_loculated_polyp(1)
					nodular_swelling(1)
					papilamotosis(0)
					papilloma(1)
					polypoid_degeneration_reinkes(1)
					vocal_fold_sulcus(1)
					vocal_fold_thickening(1)
					cyst(1)
					hemmoragic_reinkes_edema(1)
					leukoplakia(3)
					reinkes_edema(1)
					scarring(1)
					exudative_hyperkeratotic_lesions_of_epithelium(1)
					reactive_vocal_fold_lesion(1)
					vocal_fold_cyst_sub_epithelial(1)
					vocal_fold_nodules(1)
					vocal_fold_polyp(1)
					vocal_fold_scar(1)
				organic_structural_malignancy(0):
					chondroma(1)
					hypopharyngeal_tumor(1)
					microinvasive_lesion(1)
					epiglottic_carcinoma(1)
					laryngeal_tumor(1)
					carcinoma_in_situ(1)
					malignant_tumor(1)
					vocal_fold_carcinoma(1)
				organic_structural_structural_abnormality(0):
					granulation_tissue(1)
					hypokinetic_dysphonia_glottic_insufficiency(1)
					aryluxation(1)
					granuloma(1)
					fusiform_mass(1)
					left_hemilaryngectomy(1)
					partial_laryngectomy(1)
					redundant_arytenoid_mucosal_with_prolapsing_arytenoid(1)
					valleculacyste(1)
					anterior_saccular_cyst(1)
					fibroma(1)
					laryngocele(1)
					laryngeal_web(1)
					subglottal_mass(1)
					subglottal_anterior_web(1)
					subglottis_stenosis(1)
				organic_structural_vascular(0):
					hemorrhagic_polyp(1)
					bleed(0)
					hematoma(1)
					varix(1)
					chronic_hemmorage(1)
					hemorrhage(1)
					varix_and_ectasia_of_the_vocal_fold(1)
					vocal_fold_hemorrhage(1)
			organic_trauma(0):
				organic_trauma_external(0):
					arytenoid_dislocation(1)
					laryngeal_trauma(1)
					laryngeal_trauma_blunt(1)
				organic_trauma_internal(0):
					intubation_granuloma(1)
					intubation_trauma(1)
					intubation_damage(1)
					laryngeal_mucosa_trauma_chemical_and_thermal(1)
			respiratory(0)
	unclassified(1):
		abnormal_vocal_process(1)
		caustic_trauma(1)
		diffuse_mild_irregularities_of_musculomembranous_vocal_folds(1)
		dysplastic_larynx(1)
		veracosity(1)
		a_p_compression(1)
		athetoid(1)
		blunt_trauma(1)
		contact_pachyderma(1)
		cordectomy(1)
		generalized_edema_of_larynx(1)
		hypervascularization(1)
		hypokinetic_dysphonia_bilateral_vocal_fold(1)
		hypokinetic_dysphonia_dysphonia_by_chordal_groove(1)
		hypokinetic_dysphonia_laryngitis(1)
		hypotonic_dysphonia(1)
		idiopathic_neuro_disorder(1)
		interarytenoid_hyperplasia(1)
		juvenile_dysphonia(1)
		laryngopharyngeal_reflux(1)
		lesion(2)
		orofacial_dyspraxia(1)
		pachydermia(1)
		paradoxical_vocal_fold_movement(1)
		polypoid_changes(1)
		post_radiated_larynx(1)
		restriction_of_arytenoid_movement(1)
		synechia(1)
		upper_respiratory_tract_infection(0)
		ventricular_compression(6)
		ventricular_mass_on_right(1)
		choreaic_movements(1)
		dish_syndrome(1)
		dysarthria(1)
		erythema(1)
		gerd(1)
		glottal_ap_compression_mild(1)
		hyperkinetic_dysphonia_vocal_fold_prolapse(1)
		infection(0)
		lymphode_hyperplasia(1)
		mass(1)
		mesopharyngeal_tumor(1)
		post_surgery_cricoid_removal(1)
		stiffness(0)
		vox_senilis(1)
		a_p_squeezing(1)
		cystic_appearing_area(1)
		discoordinated_arytenoid_movement(1)
		dysarthrophonia(1)
		dyskinesia(1)
		dysphonia(1)
		dysplastic_dysphonia(1)
		asymmetry_of_arytenoid_movement(1)
		downs_disease(1)
		frontolateral_partial_resection(1)
		hyperkinetic_dysphonia_cordite(1)
		hyperkinetic_dysphonia_prolapse(1)
		idiopathic_dysphonia(1)
		idiopathic_laryngeal_discoordination(1)
		irritation(1)
		medial_cervical_cyst(1)
		mutation(1)
		mutation_fistula_voice(1)
		pocket_fold_hyperplasia(1)
		post_fix_for_functional_problem(1)
		post_intubation_for_seven_days(1)
		post_laser_removal_of_subglottic_web(1)
		post_vocal_cord_stripping(1)
		question_of_subglottic_masses(1)
		question_of_unknown_neurological_disorder(1)
		question_of_unknown_psychiatric_disorder(1)
		smoke_inhalation(1)
		spastic(1)
		subcordal_valley(1)
		vocal_fatigue(1)
		white_debris_patches(1)
		afrin_rhinitis(1)
		allergy_minor(1)
		anterior_mass(1)
		arytenoid(1)
		aspiration(1)
		balbuties(1)
		clutter_syndrome(0)
		cold_minor(1)
		diplophony(1)
		dysody(1)
		dysphagia(1)
		flu_minor_days_ago(1)
		head_trauma(1)
		hyperasthenia(1)
		hypokinetic_dysphonia_extraglottic_air_leak(1)
		immediate_post_surgery(1)
		internal_weakness(1)
		intubation(1)
		irregularity(1)
		major_depressive_disorder_recurrent(1)
		myasthenia(1)
		neurasthenia(0)
		non_fluency_syndrome(1)
		pocket_fold_voice(1)
		polters_syndrome(1)
		possible_subglottal_mucous_collection(1)
		post_arytenoid_adduction(1)
		post_biopsy(1)
		post_botox_injection(1)
		post_cancer_surgery(1)
		post_cancer_surgery_of_the_hypopharynx(1)
		post_laryngoplasty(1)
		post_microflap_resection(1)
		post_microflap_surgery(1)
		post_removal_of_nodular_granuloma(1)
		post_surgery_for_contact_granuloma(1)
		post_surgery_for_removal_of_teflon_granuloma(1)
		post_surgery_removal_of_keratosis_with_atypia(1)
		post_surgical_changes(1)
		post_surgical_removal_of_granulation_tissue(1)
		post_thyroplasty(1)
		post_thyroplasty_and_cricopharyngeal_myotomy(1)
		post_viral(0)
		posterior_arytenoid_lateralization_surgery(1)
		postnasal(0)
		pre_cricothyroid_approximation(1)
		prominent_lingual_tonsils(1)
		question_of_sln(1)
		rhinophony_aperta(1)
		rhinophony_clausa(1)
		rhinophony_mixta(1)
		sigmatism(1)
		singing_voice(1)
		supraglottic(1)
		thick_mucous_and_mucous_stranding(1)
		transsexual(1)
		unknown_neurological_disorder(1)
		unusual_adduction_compression(1)
		vascular_injection(1)
		velopharyngoplasty(1)
		ventricular_fold(1)
		vocal_fold(1)
		voice_disorders_undiagnosed(1)
```

In total 273 terms from DBs were automatically classified, while 0 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(27):**
	- **acute laryngitis:** acute_laryngitis
	- **amyotrophe lateralsklerose:** amyotrophic_lateral_sclerosis
	- **bilateral recurrent laryngeal nerve (rln) paralysis-peripheral:** bilateral_recurrent_laryngeal_nerve_rln_paralysis_peripheral
	- **hemmoragic reinke's edema:** hemmoragic_reinkes_edema
	- **keratosis (sometimes described as leukoplakia or erythroplasia):** leukoplakia
	- **laryngeal mucosa trauma (chemical and thermal):** laryngeal_mucosa_trauma_chemical_and_thermal
	- **laryngopharyngeal reflux:** laryngopharyngeal_reflux
	- **major depressive disorder (recurrent):** major_depressive_disorder_recurrent
	- **morbus parkinson:** parkinson_disease
	- **muscle tension dysphonia (primary):** muscle_tension
	- **muscle tension/adaptive dysphonia (secondary):** muscle_tension_adaptive
	- **nodular swelling:** nodular_swelling
	- **non-intubation related vocal fold granuloma:** non_intubation_related_vocal_fold_granuloma
	- **normal:** normal
	- **presbyphonia:** presbyphonia
	- **puberphonia:** puberphonia
	- **reactive vocal fold lesion:** reactive_vocal_fold_lesion
	- **stimmlippenpolyp:** vocal_fold_polyp
	- **sulcus vocalis:** vocal_fold_sulcus
	- **unilateral or bilateral recurrent laryngeal nerve (rln) paresis:** unilateral_or_bilateral_recurrent_laryngeal_nerve_rln_paresis
	- **unilateral recurrent laryngeal nerve (rln) paralysis:** unilateral_recurrent_laryngeal_nerve_rln_paralysis
	- **varix and ectasia of the vocal fold:** varix_and_ectasia_of_the_vocal_fold
	- **ventricular dysphonia:** ventricular_dysphonia
	- **vocal fold cyst-sub-epithelial:** vocal_fold_cyst_sub_epithelial
	- **vocal fold hemorrhage:** vocal_fold_hemorrhage
	- **vocal fold scar proper:** vocal_fold_scar
	- **voice disorders: undiagnosed or not otherwise specified (nos):** voice_disorders_undiagnosed
- **meei(168):**
	- **a-p compression (moderate):** a_p_compression
	- **a-p squeezing:** a_p_squeezing
	- **abductor spasmodic dysphonia:** abductor_spasmodic_dysphonia
	- **abnormal vocal process:** abnormal_vocal_process
	- **adductor spasmodic dysphonia:** adductor_spasmodic_dysphonia
	- **afrin rhinitis:** afrin_rhinitis
	- **anterior mass:** anterior_mass
	- **anterior saccular cyst:** anterior_saccular_cyst
	- **arytenoid:** arytenoid
	- **arytenoid dislocation:** arytenoid_dislocation
	- **aspiration:** aspiration
	- **asymmetry of arytenoid movement:** asymmetry_of_arytenoid_movement
	- **atrophic laryngitis:** atrophic_laryngitis
	- **atypical paradoxical vocal fold movement of unknown etiology:** atypical_paradoxical_vocal_fold_movement_of_unknown_etiology
	- **blunt trauma:** blunt_trauma
	- **bowing:** bowing
	- **caustic trauma:** caustic_trauma
	- **chordektomie:** cordectomy
	- **choreaic movements:** choreaic_movements
	- **chronic hemmorage:** chronic_hemmorage
	- **chronic laryngitis:** chronic_laryngitis
	- **contact granuloma:** contact_granuloma
	- **conversion aphonia:** conversion_dysphonia
	- **cricoarytenoid arthritis:** cricoarytenoid_arthritis
	- **cyst:** cyst
	- **cystic appearing area:** cystic_appearing_area
	- **diffuse mild irregularities of musculomembranous vocal folds:** diffuse_mild_irregularities_of_musculomembranous_vocal_folds
	- **discoordinated arytenoid movement:** discoordinated_arytenoid_movement
	- **dysarthria:** dysarthria
	- **dyskinesia:** dyskinesia
	- **dysphagia:** dysphagia
	- **episodic functional dysphonia:** episodic_functional_dysphonia
	- **erythema:** erythema
	- **essential tremor:** essential_tremor
	- **exudative hyperkeratotic lesions of epithelium:** exudative_hyperkeratotic_lesions_of_epithelium
	- **fusiform mass:** fusiform_mass
	- **gastric reflux:** gerd
	- **generalized edema of larynx:** generalized_edema_of_larynx
	- **gesangsstimme:** singing_voice
	- **glottal ap compression (mild):** glottal_ap_compression_mild
	- **granulation tissue:** granulation_tissue
	- **head trauma:** head_trauma
	- **hematoma:** hematoma
	- **hemmoragic reinke's edema:** hemmoragic_reinkes_edema
	- **hemorrhage:** hemorrhage
	- **hemorrhagic polyp:** hemorrhagic_polyp
	- **hyperfunction:** hyperfunction
	- **hypervascularization:** hypervascularization
	- **idiopathic dysphonia:** idiopathic_dysphonia
	- **idiopathic laryngeal discoordination:** idiopathic_laryngeal_discoordination
	- **idiopathic neuro. disorder:** idiopathic_neuro_disorder
	- **immediate post surgery:** immediate_post_surgery
	- **inflamed arytenoid:** inflamed_arytenoid
	- **inflammatory disease:** inflammatory_disease
	- **interarytenoid hyperplasia:** interarytenoid_hyperplasia
	- **intubation:** intubation
	- **intubation trauma:** intubation_trauma
	- **irregularity:** irregularity
	- **irritation:** irritation
	- **keratosis / leukoplakia:** leukoplakia
	- **keratotic reaction to polyp:** keratotic_reaction_to_polyp
	- **laryngeal trauma:** laryngeal_trauma
	- **laryngeal trauma - blunt:** laryngeal_trauma_blunt
	- **laryngeal tuberculosis:** laryngeal_tuberculosis
	- **laryngeal web:** laryngeal_web
	- **laryngocele:** laryngocele
	- **left hemilaryngectomy:** left_hemilaryngectomy
	- **lesion:** lesion
	- **lesions posterior left vocal fold:** lesion
	- **lymphode hyperplasia:** lymphode_hyperplasia
	- **malignant tumor:** malignant_tumor
	- **mass:** mass
	- **micro-cyst:** micro_cyst
	- **microinvasive lesion:** microinvasive_lesion
	- **mixed:** unclassified
	- **mixed adductor / abductor spasmodic dysphonia:** mixed_adductor_abductor_spasmodic_dysphonia
	- **morbus parkinson:** parkinson_disease
	- **multi loculated polyp:** multi_loculated_polyp
	- **multiple sclerosis:** multiple_sclerosis
	- **muscular dystrophy:** muscular_dystrophy
	- **nodular swelling:** nodular_swelling
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** allergy_minor
	- **normal voice ( cold minor ):** cold_minor
	- **normal voice ( flu minor 2 days ago ):** flu_minor_days_ago
	- **pachydermia:** pachydermia
	- **papillom:** papilloma
	- **paradoxical vocal fold movement:** paradoxical_vocal_fold_movement
	- **paralysis:** paralysis
	- **paresis:** paresis
	- **partial laryngectomy:** partial_laryngectomy
	- **pathological voice- diagnosis n/a:** pathological
	- **polypoid changes:** polypoid_changes
	- **polypoid degeneration (reinke's):** polypoid_degeneration_reinkes
	- **possible subglottal mucous collection:** possible_subglottal_mucous_collection
	- **post arytenoid adduction:** post_arytenoid_adduction
	- **post biopsy:** post_biopsy
	- **post botox injection:** post_botox_injection
	- **post cancer surgery:** post_cancer_surgery
	- **post cancer surgery of the hypopharynx:** post_cancer_surgery_of_the_hypopharynx
	- **post cva laryngeal discoordination:** post_cva_laryngeal_discoordination
	- **post fix for functional problem:** post_fix_for_functional_problem
	- **post intubation for seven days:** post_intubation_for_seven_days
	- **post irradiation:** post_radiated_larynx
	- **post laryngoplasty:** post_laryngoplasty
	- **post laser removal of subglottic web:** post_laser_removal_of_subglottic_web
	- **post microflap resection:** post_microflap_resection
	- **post microflap surgery:** post_microflap_surgery
	- **post radiation difuse edema of entire larynx:** post_radiation_difuse_edema_of_entire_larynx
	- **post radiation fibrosis:** post_radiation_fibrosis
	- **post removal of nodular granuloma:** post_removal_of_nodular_granuloma
	- **post surgery:** post_surgical_changes
	- **post surgery - removal of keratosis with atypia:** post_surgery_removal_of_keratosis_with_atypia
	- **post surgery for contact granuloma:** post_surgery_for_contact_granuloma
	- **post surgery for removal of teflon granuloma:** post_surgery_for_removal_of_teflon_granuloma
	- **post surgical removal of granulation tissue:** post_surgical_removal_of_granulation_tissue
	- **post thyroplasty:** post_thyroplasty
	- **post thyroplasty and cricopharyngeal myotomy:** post_thyroplasty_and_cricopharyngeal_myotomy
	- **post vocal cord stripping:** post_vocal_cord_stripping
	- **post-intubation submucosal edema (mild):** post_intubation_submucosal_edema_mild
	- **post-surgery -cricoid removal:** post_surgery_cricoid_removal
	- **posterior arytenoid lateralization surgery:** posterior_arytenoid_lateralization_surgery
	- **pre-cricothyroid approximation:** pre_cricothyroid_approximation
	- **pre-nodular swellings:** pre_nodular_swellings
	- **presbyphonia:** presbyphonia
	- **prominent lingual tonsils:** prominent_lingual_tonsils
	- **puberphonia:** puberphonia
	- **question of sln:** question_of_sln
	- **question of subglottic masses:** question_of_subglottic_masses
	- **question of unknown neurological disorder:** question_of_unknown_neurological_disorder
	- **question of unknown psychiatric disorder:** question_of_unknown_psychiatric_disorder
	- **redundant arytenoid mucosal with prolapsing arytenoid:** redundant_arytenoid_mucosal_with_prolapsing_arytenoid
	- **restriction of arytenoid movement:** restriction_of_arytenoid_movement
	- **scarring:** scarring
	- **smoke inhalation:** smoke_inhalation
	- **stimmlippenpolyp:** vocal_fold_polyp
	- **subcordal valley:** subcordal_valley
	- **subglottal anterior web:** subglottal_anterior_web
	- **subglottal mass:** subglottal_mass
	- **subglottis stenosis:** subglottis_stenosis
	- **sulcus vocalis:** vocal_fold_sulcus
	- **supraglottic:** supraglottic
	- **teflon granuloma:** teflon_granuloma
	- **thick mucous and mucous stranding:** thick_mucous_and_mucous_stranding
	- **transsexual:** transsexual
	- **unknown neurological disorder:** unknown_neurological_disorder
	- **unusual adduction/compression:** unusual_adduction_compression
	- **varix:** varix
	- **vascular injection:** vascular_injection
	- **ventricular compression:** ventricular_compression
	- **ventricular compression (full):** ventricular_compression
	- **ventricular compression (mild):** ventricular_compression
	- **ventricular compression (moderate):** ventricular_compression
	- **ventricular compression (severe):** ventricular_compression
	- **ventricular compression (slight):** ventricular_compression
	- **ventricular fold:** ventricular_fold
	- **ventricular mass on right:** ventricular_mass_on_right
	- **ventricular phonation:** ventricular_phonation
	- **veracosity:** veracosity
	- **vocal fatigue:** vocal_fatigue
	- **vocal fold:** vocal_fold
	- **vocal fold atrophic:** vocal_fold_atrophic
	- **vocal fold edema:** vocal_fold_edema
	- **vocal fold lesion:** vocal_fold_lesion
	- **vocal fold thickening:** vocal_fold_thickening
	- **vocal tremor:** vocal_tremor
	- **white debris/patches:** white_debris_patches
- **svd(71):**
	- **amyotrophe lateralsklerose:** amyotrophic_lateral_sclerosis
	- **aryluxation:** aryluxation
	- **balbuties:** balbuties
	- **bulbärparalyse:** bulbar_paralysis
	- **carcinoma in situ:** carcinoma_in_situ
	- **chondrom:** chondroma
	- **chordektomie:** cordectomy
	- **cyst:** cyst
	- **diplophonie:** diplophony
	- **dish-syndrom:** dish_syndrome
	- **dysarthrophonie:** dysarthrophonia
	- **dysodie:** dysody
	- **dysphonie:** dysphonia
	- **dysplastische dysphonie:** dysplastic_dysphonia
	- **dysplastischer kehlkopf:** dysplastic_larynx
	- **epiglottiskarzinom:** epiglottic_carcinoma
	- **fibrom:** fibroma
	- **frontolaterale teilresektion:** frontolateral_partial_resection
	- **funktionelle dysphonie:** functional_dysphonia
	- **gastric reflux:** gerd
	- **gesangsstimme:** singing_voice
	- **granulom:** granuloma
	- **healthy:** normal
	- **hemmoragic reinke's edema:** hemmoragic_reinkes_edema
	- **hyperasthenie:** hyperasthenia
	- **hyperfunktionelle dysphonie:** hyperfunctional_dysphonia
	- **hypofunktionelle dysphonie:** hypofunctional_dysphonia
	- **hypopharynxtumor:** hypopharyngeal_tumor
	- **hypotone dysphonie:** hypotonic_dysphonia
	- **internusschwäche:** internal_weakness
	- **intubation trauma:** intubation_trauma
	- **intubationsgranulom:** intubation_granuloma
	- **juvenile dysphonie:** juvenile_dysphonia
	- **kehlkopftumor:** laryngeal_tumor
	- **kontaktpachydermie:** contact_pachyderma
	- **laryngitis:** laryngitis
	- **laryngocele:** laryngocele
	- **leukoplakie:** leukoplakia
	- **mediale halscyste:** medial_cervical_cyst
	- **mesopharynxtumor:** mesopharyngeal_tumor
	- **monochorditis:** monochorditis
	- **morbus down:** downs_disease
	- **morbus parkinson:** parkinson_disease
	- **mutatio:** mutation
	- **mutationsfistelstimme:** mutation_fistula_voice
	- **n. laryngeus superior läsion:** superior_laryngeal_nerve_lesion
	- **n. laryngeus superior neuralgie:** superior_laryngeal_nerve_neuralgia
	- **non-fluency-syndrom:** non_fluency_syndrome
	- **orofaciale dyspraxie:** orofacial_dyspraxia
	- **papillom:** papilloma
	- **phonasthenie:** myasthenia
	- **phonationsknötchen:** phonation_nodules
	- **poltersyndrom:** polters_syndrome
	- **psychogene aphonie:** psychogenic_aphonia
	- **psychogene dysphonie:** psychogenic_dysphonia
	- **psychogene mikrophonie:** psychogenic_microphony
	- **rekurrensparese:** recurrent_paralysis
	- **rhinophonie aperta:** rhinophony_aperta
	- **rhinophonie clausa:** rhinophony_clausa
	- **rhinophonie mixta:** rhinophony_mixta
	- **sigmatismus:** sigmatism
	- **spasmodische dysphonie:** spasmodic_dysphonia
	- **stimmlippenkarzinom:** vocal_fold_carcinoma
	- **stimmlippenpolyp:** vocal_fold_polyp
	- **synechie:** synechia
	- **taschenfaltenhyperplasie:** pocket_fold_hyperplasia
	- **taschenfaltenstimme:** pocket_fold_voice
	- **valleculacyste:** valleculacyste
	- **velopharyngoplastik:** velopharyngoplasty
	- **vox senilis:** vox_senilis
	- **zentral-laryngale bewegungsstörung:** central_laryngeal_movement_disorder
- **torgo(2):**
	- **dysarthria:** dysarthria
	- **normal:** normal
- **uaspeech(4):**
	- **athetoid:** athetoid
	- **mixed:** unclassified
	- **normal:** normal
	- **spastic:** spastic
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(24):**
	- **healthy:** normal
	- **hyperkinetic dysphonia:** hyperkinetic_dysphonia
	- **hyperkinetic dysphonia  (rigid vocal fold):** hyperkinetic_dysphonia_rigid_vocal_fold
	- **hyperkinetic dysphonia (adduction deficit):** hyperkinetic_dysphonia_adduction_deficit
	- **hyperkinetic dysphonia (cordite):** hyperkinetic_dysphonia_cordite
	- **hyperkinetic dysphonia (nodule):** hyperkinetic_dysphonia_nodule
	- **hyperkinetic dysphonia (polyps):** hyperkinetic_dysphonia_polyps
	- **hyperkinetic dysphonia (prolapse):** hyperkinetic_dysphonia_prolapse
	- **hyperkinetic dysphonia (reinke's edema):** hyperkinetic_dysphonia_reinkes_edema
	- **hyperkinetic dysphonia (vocal fold nodules):** hyperkinetic_dysphonia_vocal_fold_nodules
	- **hyperkinetic dysphonia (vocal fold paralysis):** hyperkinetic_dysphonia_vocal_fold_paralysis
	- **hyperkinetic dysphonia (vocal fold prolapse):** hyperkinetic_dysphonia_vocal_fold_prolapse
	- **hypokinetic dysphonia:** hypokinetic_dysphonia
	- **hypokinetic dysphonia (adduction deficit):** hypokinetic_dysphonia_adduction_deficit
	- **hypokinetic dysphonia (bilateral vocal fold):** hypokinetic_dysphonia_bilateral_vocal_fold
	- **hypokinetic dysphonia (conversion dysphonia):** hypokinetic_dysphonia_conversion_dysphonia
	- **hypokinetic dysphonia (dysphonia by chordal groove):** hypokinetic_dysphonia_dysphonia_by_chordal_groove
	- **hypokinetic dysphonia (extraglottic air leak):** hypokinetic_dysphonia_extraglottic_air_leak
	- **hypokinetic dysphonia (glottic insufficiency):** hypokinetic_dysphonia_glottic_insufficiency
	- **hypokinetic dysphonia (laryngitis):** hypokinetic_dysphonia_laryngitis
	- **hypokinetic dysphonia (presbiphonia):** hypokinetic_dysphonia_presbiphonia
	- **hypokinetic dysphonia (spasmodic dysphonia):** hypokinetic_dysphonia_spasmodic_dysphonia
	- **hypokinetic dysphonia (vocal fold paralysis):** hypokinetic_dysphonia_vocal_fold_paralysis
	- **reflux laryngitis:** reflux_laryngitis

The following aliases were missing:
- **muscle_tension(1):** ['primary muscle tension dysphonia']

And the following number of terms were left unmatched across the different databases:
- **avfad(0):** []
- **meei(0):** []
- **svd(0):** []
- **torgo(0):** []
- **uaspeech(0):** []
- **uncommon_voice(0):** []
- **voiced(0):** []

### Zaim_2023
```
	pathological(2):
		non_structural_dysphonia(0):
			bilateral_vocal_fold_palsy(0)
			presbylarynx(0)
			primary_muscle_tension_dysphonia(1)
			psychogenic_dysphonia(1)
			puberphonia(1)
			spasmodic_dysphonia(1)
			unilateral_vocal_fold_palsy(0)
			voice_tremor(1)
		structural_dysphonia(0):
			benign_lesion(1)
			funcal_laryngitis(0)
			inflammatory_lesion(0)
			laryngeal_amyloidosis(0)
			laryngeal_perichondritis(0)
			laryngitis(1)
			laryngopharyngeal_reflux(1)
			malignant_lesion(0)
			premalignant_lesion(0)
			recurrent_respiratory_papillomatosis(0)
			sulcus_vocalis(1)
			tuberculosis_laryngitis(1)
			vocal_fold_cyst(1)
			vocal_fold_edema(1)
			vocal_fold_nodule(1)
			vocal_fold_polyp(1)
	normal(7)
	unclassified(1)
```

In total 24 terms from DBs were automatically classified, while 249 were left unclassified.

The diagnostic terms were allocated as following:
- **avfad(7):**
	- **laryngopharyngeal reflux:** laryngopharyngeal_reflux
	- **muscle tension dysphonia (primary):** primary_muscle_tension_dysphonia
	- **nodular swelling:** vocal_fold_nodule
	- **normal:** normal
	- **puberphonia:** puberphonia
	- **stimmlippenpolyp:** vocal_fold_polyp
	- **sulcus vocalis:** sulcus_vocalis
- **meei(17):**
	- **cyst:** vocal_fold_cyst
	- **essential tremor:** voice_tremor
	- **gesangsstimme:** normal
	- **laryngeal tuberculosis:** tuberculosis_laryngitis
	- **lesion:** benign_lesion
	- **mixed:** unclassified
	- **nodular swelling:** vocal_fold_nodule
	- **normal:** normal
	- **normal voice:** normal
	- **normal voice ( allergy minor ):** normal
	- **normal voice ( cold minor ):** normal
	- **normal voice ( flu minor 2 days ago ):** normal
	- **pathological voice- diagnosis n/a:** pathological
	- **puberphonia:** puberphonia
	- **stimmlippenpolyp:** vocal_fold_polyp
	- **sulcus vocalis:** sulcus_vocalis
	- **vocal fold edema:** vocal_fold_edema
- **svd(7):**
	- **cyst:** vocal_fold_cyst
	- **gesangsstimme:** normal
	- **healthy:** normal
	- **laryngitis:** laryngitis
	- **psychogene dysphonie:** psychogenic_dysphonia
	- **spasmodische dysphonie:** spasmodic_dysphonia
	- **stimmlippenpolyp:** vocal_fold_polyp
- **torgo(1):**
	- **normal:** normal
- **uaspeech(2):**
	- **mixed:** unclassified
	- **normal:** normal
- **uncommon_voice(2):**
	- **normal:** normal
	- **pathological:** pathological
- **voiced(1):**
	- **healthy:** normal

The following aliases were missing:
- **normal(7):** ['gesangsstimme', 'singing_voice', 'normal voice (singing training)', 'singing training', 'singing voice', "singer's voice", 'sängerstimme']

And the following number of terms were left unmatched across the different databases:
- **avfad(20):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', "hemmoragic reinke's edema", 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'major depressive disorder (recurrent)', 'morbus parkinson', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'reactive vocal fold lesion', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'unilateral recurrent laryngeal nerve (rln) paralysis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(151):** ['a-p compression (moderate)', 'a-p squeezing', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'generalized edema of larynx', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', "hemmoragic reinke's edema", 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'morbus parkinson', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold lesion', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(64):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'carcinoma in situ', 'chondrom', 'chordektomie', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'epiglottiskarzinom', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'granulom', "hemmoragic reinke's edema", 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'morbus parkinson', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene mikrophonie', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'stimmlippenkarzinom', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(23):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold paralysis)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'hypokinetic dysphonia (vocal fold paralysis)', 'reflux laryngitis']

