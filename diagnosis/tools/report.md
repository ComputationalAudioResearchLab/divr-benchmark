# Diagnostic Terms Usage

## Counting terms across DBs
7 databases were used in this study, namely ['avfad', 'meei', 'svd', 'torgo', 'uaspeech', 'uncommon_voice', 'voiced'].
These DBs in total had 306 diagnostic terms.
We duplicated terms based on typographical variations (e.g. cyst vs cysts, reinke's edema vs reinke edema), translations (e.g. leukoplakia vs leukoplakie), and synonymous terms (e.g. not sure vs unknown, functional voice disorder vs functional dysphonia). After deduplication wer have 276 diagnostic terms across the 7 databases. The different DBs had the following number of diagnostic terms: {'meei': 170, 'svd': 71, 'avfad': 27, 'voiced': 24, 'uaspeech': 4, 'torgo': 2, 'uncommon_voice': 2}.

## Categorisation of other classification systems

### Benba_2017
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
	normal(2)
	unclassified(1)


The diagnostic terms were allocated as following:
- **avfad(2):**
	- **morbus parkinson:** parkinsons
	- **normal:** normal
- **meei(5):**
	- **essential tremor:** essential_tremors
	- **mixed:** unclassified
	- **morbus parkinson:** parkinsons
	- **normal:** normal
	- **pathological voice- diagnosis n/a:** pathological
- **svd(2):**
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
- **essential_tremors(1):** ['essential tremor']
- **normal(1):** ['without dysarthria']
- **parkinsons(5):** ['morbus parkinson', 'parkinson_disease', 'parkinson disease', "parkinson's disease", 'parkinsons syndrome']
- **unclassified(4):** ['mixed', 'other', 'others', 'unknown']

And the following number of terms were left unmatched across the different databases:
- **avfad(25):** ['acute laryngitis', 'amyotrophe lateralsklerose', 'bilateral recurrent laryngeal nerve (rln) paralysis-peripheral', 'keratosis (sometimes described as leukoplakia or erythroplasia)', 'laryngeal mucosa trauma (chemical and thermal)', 'laryngopharyngeal reflux', 'major depressive disorder (recurrent)', 'muscle tension dysphonia (primary)', 'muscle tension/adaptive dysphonia (secondary)', 'non-intubation related vocal fold granuloma', 'presbyphonia', 'puberphonia', 'reactive vocal fold lesion', 'reinke ödem', 'stimmlippenpolyp', 'sulcus vocalis', 'unilateral or bilateral recurrent laryngeal nerve (rln) paresis', 'unilateral recurrent laryngeal nerve (rln) paralysis', 'varix and ectasia of the vocal fold', 'ventricular dysphonia', 'vocal fold cyst-sub-epithelial', 'vocal fold hemorrhage', 'vocal fold nodules', 'vocal fold scar proper', 'voice disorders: undiagnosed or not otherwise specified (nos)']
- **meei(165):** ['a-p compression (moderate)', 'a-p squeezing', 'a-p squeezing (severe)', 'abductor spasmodic dysphonia', 'abnormal vocal process', 'adductor spasmodic dysphonia', 'afrin rhinitis', 'anterior mass', 'anterior saccular cyst', 'arytenoid', 'arytenoid dislocation', 'aspiration', 'asymmetry of arytenoid movement', 'atrophic laryngitis', 'atypical paradoxical vocal fold movement of unknown etiology', 'blunt trauma', 'bowing', 'caustic trauma', 'chordektomie', 'choreaic movements', 'chronic hemmorage', 'chronic laryngitis', 'contact granuloma', 'conversion aphonia', 'cricoarytenoid arthritis', 'cyst', 'cystic appearing area', 'diffuse mild irregularities of musculomembranous vocal folds', 'discoordinated arytenoid movement', 'dysarthria', 'dyskinesia', 'dysphagia', 'episodic functional dysphonia', 'erythema', 'exudative hyperkeratotic lesions of epithelium', 'fusiform mass', 'gastric reflux', 'generalized edema of larynx', 'gesangsstimme', 'glottal ap compression (mild)', 'granulation tissue', 'head trauma', 'hematoma', "hemmoragic reinke's edema", 'hemorrhage', 'hemorrhagic polyp', 'hyperfunction', 'hypervascularization', 'idiopathic dysphonia', 'idiopathic laryngeal discoordination', 'idiopathic neuro. disorder', 'immediate post surgery', 'inflamed arytenoid', 'inflammatory disease', 'interarytenoid hyperplasia', 'intubation', 'intubation trauma', 'irregularity', 'irritation', 'keratosis / leukoplakia', 'keratotic reaction to polyp', 'laryngeal trauma', 'laryngeal trauma - blunt', 'laryngeal tuberculosis', 'laryngeal web', 'laryngocele', 'left hemilaryngectomy', 'lesion', 'lesions posterior left vocal fold', 'lymphode hyperplasia', 'malignant tumor', 'mass', 'micro-cyst', 'microinvasive lesion', 'mixed adductor / abductor spasmodic dysphonia', 'multi loculated polyp', 'multiple sclerosis', 'muscular dystrophy', 'nodular swelling', 'normal voice', 'normal voice ( allergy minor )', 'normal voice ( cold minor )', 'normal voice ( flu minor 2 days ago )', 'pachydermia', 'papillom', 'paradoxical vocal fold movement', 'paralysis', 'paresis', 'partial laryngectomy', 'polypoid changes', "polypoid degeneration (reinke's)", 'possible subglottal mucous collection', 'post arytenoid adduction', 'post biopsy', 'post botox injection', 'post cancer surgery', 'post cancer surgery of the hypopharynx', 'post cva laryngeal discoordination', 'post fix for functional problem', 'post intubation for seven days', 'post irradiation', 'post laryngoplasty', 'post laser removal of subglottic web', 'post microflap resection', 'post microflap surgery', 'post radiation difuse edema of entire larynx', 'post radiation fibrosis', 'post removal of nodular granuloma', 'post surgery', 'post surgery - removal of keratosis with atypia', 'post surgery for contact granuloma', 'post surgery for removal of teflon granuloma', 'post surgical removal of granulation tissue', 'post thyroplasty', 'post thyroplasty and cricopharyngeal myotomy', 'post vocal cord stripping', 'post-intubation submucosal edema (mild)', 'post-surgery -cricoid removal', 'posterior arytenoid lateralization surgery', 'pre-cricothyroid approximation', 'pre-nodular swellings', 'presbyphonia', 'prominent lingual tonsils', 'puberphonia', 'question of sln', 'question of subglottic masses', 'question of unknown neurological disorder', 'question of unknown psychiatric disorder', 'redundant arytenoid mucosal with prolapsing arytenoid', 'restriction of arytenoid movement', 'scarring', 'smoke inhalation', 'stimmlippenpolyp', 'subcordal valley', 'subglottal anterior web', 'subglottal mass', 'subglottis stenosis', 'sulcus vocalis', 'supraglottic', 'teflon granuloma', 'thick mucous and mucous stranding', 'transsexual', 'unknown neurological disorder', 'unusual adduction/compression', 'varix', 'vascular injection', 'ventricular compression', 'ventricular compression (full)', 'ventricular compression (mild)', 'ventricular compression (moderate)', 'ventricular compression (severe)', 'ventricular compression (slight)', 'ventricular fold', 'ventricular mass on right', 'ventricular phonation', 'veracosity', 'vocal fatigue', 'vocal fold', 'vocal fold atrophic', 'vocal fold edema', 'vocal fold lesion', 'vocal fold nodules', 'vocal fold thickening', 'vocal tremor', 'white debris/patches']
- **svd(69):** ['amyotrophe lateralsklerose', 'aryluxation', 'balbuties', 'bulbärparalyse', 'carcinoma in situ', 'chondrom', 'chordektomie', 'cyst', 'diplophonie', 'dish-syndrom', 'dysarthrophonie', 'dysodie', 'dysphonie', 'dysplastische dysphonie', 'dysplastischer kehlkopf', 'epiglottiskarzinom', 'fibrom', 'frontolaterale teilresektion', 'funktionelle dysphonie', 'gastric reflux', 'gesangsstimme', 'granulom', 'hyperasthenie', 'hyperfunktionelle dysphonie', 'hypofunktionelle dysphonie', 'hypopharynxtumor', 'hypotone dysphonie', 'internusschwäche', 'intubation trauma', 'intubationsgranulom', 'juvenile dysphonie', 'kehlkopftumor', 'kontaktpachydermie', 'laryngitis', 'laryngocele', 'leukoplakie', 'mediale halscyste', 'mesopharynxtumor', 'monochorditis', 'morbus down', 'mutatio', 'mutationsfistelstimme', 'n. laryngeus superior läsion', 'n. laryngeus superior neuralgie', 'non-fluency-syndrom', 'orofaciale dyspraxie', 'papillom', 'phonasthenie', 'phonationsknötchen', 'poltersyndrom', 'psychogene aphonie', 'psychogene dysphonie', 'psychogene mikrophonie', 'reinke ödem', 'rekurrensparese', 'rhinophonie aperta', 'rhinophonie clausa', 'rhinophonie mixta', 'sigmatismus', 'spasmodische dysphonie', 'stimmlippenkarzinom', 'stimmlippenpolyp', 'synechie', 'taschenfaltenhyperplasie', 'taschenfaltenstimme', 'valleculacyste', 'velopharyngoplastik', 'vox senilis', 'zentral-laryngale bewegungsstörung']
- **torgo(1):** ['dysarthria']
- **uaspeech(2):** ['athetoid', 'spastic']
- **uncommon_voice(0):** []
- **voiced(23):** ['hyperkinetic dysphonia', 'hyperkinetic dysphonia  (rigid vocal fold)', 'hyperkinetic dysphonia (adduction deficit)', 'hyperkinetic dysphonia (cordite)', 'hyperkinetic dysphonia (nodule)', 'hyperkinetic dysphonia (polyps)', 'hyperkinetic dysphonia (prolapse)', "hyperkinetic dysphonia (reinke's edema)", 'hyperkinetic dysphonia (vocal fold nodules)', 'hyperkinetic dysphonia (vocal fold paralysis)', 'hyperkinetic dysphonia (vocal fold prolapse)', 'hypokinetic dysphonia', 'hypokinetic dysphonia (adduction deficit)', 'hypokinetic dysphonia (bilateral vocal fold)', 'hypokinetic dysphonia (conversion dysphonia)', 'hypokinetic dysphonia (dysphonia by chordal groove)', 'hypokinetic dysphonia (extraglottic air leak)', 'hypokinetic dysphonia (glottic insufficiency)', 'hypokinetic dysphonia (laryngitis)', 'hypokinetic dysphonia (presbiphonia)', 'hypokinetic dysphonia (spasmodic dysphonia)', 'hypokinetic dysphonia (vocal fold paralysis)', 'reflux laryngitis']

### Compton_2022
### Cordeiro_2015
### daSilvaMoura_2024
### deMoraesLimaMarinus_2013
### FEMH_2018
### Kim_2024
### Sztaho_2018
### Tsui_2018
### USVAC_2025
### Zaim_2023
