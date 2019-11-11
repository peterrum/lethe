# Script pour lancer à répétition le step-60 en modifiant le fichier de paramètres
# Antoine Rincent, 2019

# Tous les import pour les fonctions utilisées
import sys
import os
import os.path
import subprocess
from collections import Counter
from natsort import natsorted, ns, humansorted, natsort_keygen
import time
import termtables as tt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score

# Fonction custom créées

# Fonction du menu d'option
def OptionMenu():
	print("\n\n**************************************************************************")
	print("Sélectionnez une des options suivantes : \n")
	print("1 : Paramétrer les tests à faire")
	print("2 : Afficher la liste des tests qui seront effectués")
	print("3 : Paramétrer l'appel du step-60, actuellement : "+ appel)
	print("4 : Changer le nom du fichier de paramètres recherché, actuellement : "+parameter_file)
	print("5 : Changer le délai de timeout, actuellement : "+str(delai)+"s")
	print("6 : Lancer les calculs")
	print("7 : Imprimer l'output des simulations, "+str(len(total_outputs))+" outputs en mémoire")
	print("8 : Sauvegarder les résultats en mémoire dans un fichier .txt")
	print("9 : Lire les résultats d'un fichier .txt")
	print("10 : Créer graphiques des résultats")
	print("\n000 : Quitter le programme")
	print("**************************************************************************\n")

	global option
	option = input()

# Lecture du fichier de paramètres
def ReadParameters():
	global Parameters
	global Dimensions
	global embedded_refinement_line
	global embedded_refinement_text
	global embedding_refinement_line
	global embedding_refinement_text
	global embedded_fedegree_line
	global embedded_fedegree_text
	global embedding_fedegree_line
	global embedding_fedregree_text
	global parameter_file

	with open(parameter_file) as f:
		Parameters = f.readlines()
	Parameters = [x.strip() for x in Parameters] 

	Dimensions = Parameters[2][-2]

	# Trouver les lignes où se trouvent les paramètres à modifier
	for line in range(0,len(Parameters),1):
		if 'set Initial embedded space refinement' in Parameters[line]:
			embedded_refinement_line = line
			embedded_refinement_text = Parameters[line][:-1]
		elif 'set Initial embedding space refinement' in Parameters[line]:
			embedding_refinement_line = line
			embedding_refinement_text = Parameters[line][:-1]
		elif 'set Embedded space finite element degree' in Parameters[line]:
			embedded_fedegree_line = line
			embedded_fedegree_text = Parameters[line][:-1]
		elif 'set Embedding space finite element degree' in Parameters[line]:
			embedding_fedegree_line = line
			embedding_fedregree_text = Parameters[line][:-1]
	None

# Fonction demander FE degree pour embedding
def Obtenir_FEdeg_embedding():
	global embedding_fedregree_value
	print("Quelle valeur du Embedding space finite element degree voulez vous? : ")
	while True:
		choix = input()
		if choix == '':
			print("Vous devez mettre une valeur")
		elif int(choix)>0:
			embedding_fedregree_value = int(choix)
			break
		else:
			print("Prière de donner un input valide")

# Fonction demander FE degree pour embedded
def Obtenir_FEdeg_embedded():
	global embedded_fedregree_value
	print("Quelle valeur du Embedded space finite element degree voulez vous? : ")
	while True:
		choix = input()
		if choix == '':
			print("Vous devez mettre une valeur")
		elif int(choix)>0:
			embedded_fedregree_value = int(choix)
			break
		else:
			print("Prière de donner un input valide")


# Fonction demander le raffinement de la maille embedded
def Obtenir_Embedded_raffinement():
	global embedded_refinement_value
	print("Quelle valeur du Initial embedded space refinement voulez vous? : ")
	while True:
		choix = input()
		if choix == '':
			print("Vous devez mettre une valeur")
		elif int(choix)>0:
			embedded_refinement_value = int(choix)
			break
		else:
			print("Prière de donner un input valide")







# Script en tant que tel
appel = '../step-60'
parameter_file = 'parameters.prm'
delai = int(0.5*3600)
total_outputs = []
ReadParameters()
filepath = os.getcwd()
test_lances = 0
titles = [0]*4
table_resultats = []

print("Créé par Antoine Rincent en 2019 pour Deal.II")
# Boucle qui sert à garder le "menu" actif
while True:
	OptionMenu()
	# Paramétrer les tests à faire
	if option == "1":
		print(option+" : Paramétrer les tests à faire")

		# Choix du paramètre à contrôler
		print("Quel paramètre voulez-vous faire varier?")
		print("1 : Raffinement du domaine embedding (total)")
		print("2 : Raffinement du domaine embedded (interne)")
		print("3 : Ordre des éléments finis du domaine embedding (total)")
		print("4 : Ordre des éléments finis du domaine embedded (interne)")
		while True:
			choix = input()
			if choix == "1":
				independant = 1
				break
			elif choix == "2":
				independant = 2
				break
			elif choix == "3":
				independant = 3
				break
			elif choix == "4":
				independant = 4
				break
			else:
				print("Prière de choisir un chiffre entre 1 et 4")

		# Paramétrisation de la variable indépendante
		print("Quelles valeurs voulez-vous lui donner? Donner autant de valeurs que voulues puis 000 pour arrêter : ")
		print("NB aucun test n'est effectué sur la validitié de l'input fourni!")
		table_valeurs = []
		while True:
			valeur_independante = [0]*4
			choix = input()
			if choix == "000":
				break
			elif choix == '':
				print("Vous devez mettre une valeur")
			elif int(choix)>0 :
				value = int(choix)
				valeur_independante[0] = value
				table_valeurs.append(valeur_independante)
			else:
				print("Prière de donner un input valide")

		# Tri naturel (comme Windows) de la liste des valeurs sélectionnées
		natsort_key = natsort_keygen(alg=ns.FLOAT)
		table_valeurs.sort(key=natsort_key)

		# Paramétrer les autres paramètres
		print("Quel écart voulez-vous avoir entre le raffinement du domaine embedding et embedded?")
		while True:
			choix = input()
			if int(choix)>0:
				ecart_embedding_embedded = int(choix)
				break
			elif choix == '':
				print("Vous devez mettre une valeur")
			else:
				print("Prière de donner un input valide")

		if independant == 1:
			col_embedding = 0
			col_embedded = 1
			col_fedeg_embedding = 2
			col_fedeg_embedded = 3
			Obtenir_FEdeg_embedding()
			Obtenir_FEdeg_embedded()
		elif independant == 2:
			col_embedding = 1
			col_embedded = 0
			col_fedeg_embedding = 2
			col_fedeg_embedded = 3
			Obtenir_FEdeg_embedding()
			Obtenir_FEdeg_embedded()
		elif independant == 3:
			col_embedding = 1
			col_embedded = 2
			col_fedeg_embedding = 0
			col_fedeg_embedded = 3
			Obtenir_Embedded_raffinement()
			Obtenir_FEdeg_embedded()
		elif independant == 4:
			col_embedding = 1
			col_embedded = 2
			col_fedeg_embedding = 3
			col_fedeg_embedded = 0
			Obtenir_Embedded_raffinement()
			Obtenir_FEdeg_embedding()

		# Remplir la table des valeurs de tests
		for test in range(0, len(table_valeurs), 1):
			# En premier, paramétrer embedded_refinement_value
			if independant == 2:
				table_valeurs[test][col_embedding] = table_valeurs[test][0] - ecart_embedding_embedded
			elif ((independant == 3) or (independant == 4)):
				table_valeurs[test][col_embedding] = embedded_refinement_value

			# Ensuite, paramétrer embedding_refinement_value
			if independant != 2:
				table_valeurs[test][col_embedding+1] = table_valeurs[test][col_embedding] + ecart_embedding_embedded

			# Ensuite, paramétrer FE degrees embedding
			if independant != 3:
				table_valeurs[test][col_fedeg_embedding] = embedding_fedregree_value

			# Enfin, paramétrer FE degrees embedded
			if independant != 4:
				table_valeurs[test][col_fedeg_embedded] = embedded_fedregree_value
				
		# Setup les titres de la table
		titles = [0]*4
		titles[col_embedding] = "Embedding Ref."
		titles[col_fedeg_embedded] = "Embedded FE Degree"
		titles[col_fedeg_embedding] = "Embedding FE Degree"
		titles[col_embedded] = "Embedded Ref."
					   

	# Afficher la liste de tests à effectuer
	elif option == "2":
		print(option + " : Afficher la liste des tests qui seront effectués\n")
		tableau = tt.to_string(table_valeurs, header=titles, alignment="cccc", padding = (0,1), style=tt.styles.ascii_thin_double)
		print(tableau)


	# Paramétrer appel step-60
	elif option == "3":
		print(option + " : Paramétrer l'appel du step-60, actuellement : "+ appel)
		print("\nSVP écrire l'appel à faire dans la console à partir du dossier actuel pour lancer le step-60 : ")
		appel = input()


	# Paramétrer l'appel du fichier paramètres
	elif option == "4":
		print(option+" : Changer le nom du fichier de paramètres recherché, actuellement : "+parameter_file)
		print("\nSVP écrire le nom du fichier paramètres ainsi que sa terminaison : ")
		temp = input()

		if os.path.isfile(os.path.join(filepath,temp)):
			parameter_file = temp
		else:
			print("Fichier non trouvé dans le répertoire actuel, changement pas appliqué.")


	# Paramétrer le délai de timeout de la console
	elif option == "5":
		print(option+" : Changer le délai de timeout, actuellement : "+str(delai))
		print("Donnez le temps de délai voulu en heures, si vous ne voulez aucun délai mettez 0")
		temp = input()
		if temp == "0":
			delai = None
		else:
			delai = int(float(temp)*3600)


	# Lancer le step-60
	elif option == "6":
		print(option + " : Lancer les calculs")

		print("Script parameters.prm paramétré pour simulations en "+Dimensions+"D")

		# Boucle pour aller de test en test
		table_resultats = [0]*len(table_valeurs)
		resultats_titres = ["Test #"] + titles + ["Erreur L2", "CPU Time", "Ratio", "Steps", "Error value"]

		total_outputs = [0]*len(table_valeurs)

		for simulation in range(0, len(table_valeurs), 1):
			# Modifier le fichier paramètres
			Parameters[embedded_refinement_line] = embedded_refinement_text + str(table_valeurs[simulation][col_embedded])
			Parameters[embedding_refinement_line] = embedding_refinement_text + str(table_valeurs[simulation][col_embedding])
			Parameters[embedded_fedegree_line] = embedded_fedegree_text + str(table_valeurs[simulation][col_fedeg_embedded])
			Parameters[embedding_fedegree_line] = embedding_fedregree_text + str(table_valeurs[simulation][col_fedeg_embedding])

			# Écriture dans parameters
			with open(parameter_file, 'w') as f:
				for item in Parameters:
					f.write("%s\n" % item)


			# Lancer avec subprocess
			cp = subprocess.run(appel, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=delai)
			if cp.returncode == 0:
				print("Exécution du test #"+str(simulation+1)+" réussie.")
				test_lances = 1

				# Traitement et stockage output du step-60
				Output = cp.stdout.split('\n')
				total_outputs[simulation] = Output

				# Extraction résultats
				for line in range(0,len(Output),1):
					if 'DEAL::Embedding minimal diameter' in Output[line]:
						Ratio = Output[line].split("ratio: ")[1]
					if 'DEAL:cg::Convergence step ' in Output[line]:
						Step = Output[line].split(" ")[2]
						ErrorValue = Output[line].split(" ")[4]
					if 'L2 error = ' in Output[line]:
						L2error = Output[line].split(" = ")[1]
					if "Total CPU time elapsed since start" in Output[line]:
						CPUTime = Output[line].split("|")[2].strip()[:-1]

				# Stockage résultats dans table_resultats
				table_resultats[simulation] = [int(simulation+1)] + table_valeurs[simulation] + [L2error, CPUTime, Ratio, Step, ErrorValue]


			else:
				print("Erreur lors de l'exécution, message d'erreur fourni : ")
				print(cp.stderr)

		# Impression résultats dans tableau pour tester fonctionnement
		tableau = tt.to_string(table_resultats, header=resultats_titres, alignment="cccccccccc", padding = (0,1), style=tt.styles.ascii_thin_double)
		print(tableau)


	# Imprimer l'output des résultats
	elif option == "7":
		print(option+" : Imprimer l'output des simulations, "+str(len(total_outputs))+" outputs en mémoire")
		if test_lances == 1:
			print("Voulez-vous tous les imprimer (1) ou un seul (2) ?")
			while True:
				choix = input()

				# Si tous à imprimer
				if choix == "1":
					for output_to_print in range(0, len(total_outputs),1):
						print('\nSimulation #'+int(output_to_print+1))
						for line in range(0,len(total_outputs[output_to_print]),1):
							print(total_outputs[output_to_print][line])
					break

				# si un seul à imprimer
				if choix == "2":
					print("Lequel voulez-vous print?")
					while True:
						choix = input()
						if (int(choix) >= 0 and int(choix)<len(total_outputs)):
							for line in range(0,len(total_outputs[int(choix)]),1):
								print(total_outputs[int(choix)][line])
							break
				
					break
		else:
			print("Aucun test n'a été lancé!")


	# Sauvegarder les résultats dans un fichier .txt
	elif option == "8":
		print(option + " : Sauvegarder les résultats en mémoire dans un fichier .txt")
		print('/nWork in progress')
		if test_lances == 1:
			# Pour nommer le fichier créée
			date = time.strftime("%d_%m")
			hour = time.strftime("%H_%M_%S")

			csvfile = open("Resultats "+date+" "+hour+".txt", "w")
			csvfile.write(tableau)
			csvfile.close()


		else:
			print("Aucun test n'a été lancé!")


	# Lire les résultats d'un fichier .txt
	elif option == "9":
		print(option + " : Lire les résultats d'un fichier .txt")
		
		# Imprimer la liste de fichiers .txt présente
		print("Dossier actuel : " + os.getcwd())
		final = ".txt"
		documents = os.listdir(os.getcwd())
		fichierspossibles = []
		numero = 1
		for text in documents:
			if final in text:
				fichierspossibles = fichierspossibles + [text]
				print(str(numero)+". "+text)
				numero += 1

		# Sélectionner un fichier parmis la liste
		print("Entrer le numéro du fichier .txt à importer")
		while True:
			numero = input()
			if int(numero) > (len(fichierspossibles)):
				print("SVP entrer un numéro inférieur ou égal à " + str(len(fichierspossibles)))
			elif int(numero) < 1:
				print("SVP entrer un numéro supérieur ou égual à 1")
			else:
				print(fichierspossibles[int(numero)-1] + " sera traité.")
				fichier = fichierspossibles[int(numero)-1]
				break


		# Lire le document fourni
		# Stockage du document dans texte
		with open(fichier,'r') as file:
			texte = file.readlines()
		texte = [x.strip() for x in texte]

		# Extraction des titres dans texte_titres
		texte_titres = texte[1].split('|')
		texte_titres = [x.strip() for x in texte_titres]
		del texte_titres[-1]
		del texte_titres[0]

		# Enlever les lignes séparant les valeurs
		for line in range(len(texte), 0,-1):
			if (line%2)==1:
				del texte[line-1]
		# Enlever ligne du titre
		del texte[0]

		# Stocker dans table_resultats_imported
		table_resultats_imported = []
		for line in range(len(texte)):
			texte[line] = texte[line].split('|')
			texte[line] = [x.strip(' ') for x in texte[line]]
			del texte[line][-1]
			del texte[line][0]

			# Updater le numéro du test en fonction de ce qui y est déjà
			texte[line][0] = str(len(table_resultats)+line+1)

		# Ajouter un tri des ordres de colonnes?
		if len(table_resultats)==0:
		
			for titre in range(len(texte_titres)):
				if "Embedding Ref." in texte_titres[titre]:
					col_embedding = int(titre-1)
				elif "Embedded FE Degree" in texte_titres[titre]:
					col_fedeg_embedded = int(titre-1)
				elif "Embedding FE Degree" in texte_titres[titre]:
					col_fedeg_embedding = int(titre-1)
				elif "Embedded Ref." in texte_titres[titre]:
					col_embedded = int(titre-1)
		
		
		else:
			print("Il est ici assumé que les colonnes Embedding Ref., ..., Embedded FE Degree, sont dans le même ordre dans les tests roulés et dans le fichier importé")

		# Ajouter les nouvelles colonnes aux résultats actuels
		if len(table_resultats)==0:
			table_resultats = texte
			resultats_titres = texte_titres

			table_valeurs = [0]*len(texte)
			for line in range(len(texte)):
				table_valeurs[line] = texte[line][1:5]
		else:
			for line in range(len(texte)):
				table_resultats.append(texte[line])

				table_valeurs.append(texte[line][1:5])

		test_lances = 1

		# Impression résultats dans tableau pour tester fonctionnement
		tableau = tt.to_string(table_resultats, header=resultats_titres, alignment="cccccccccc", padding = (0,1), style=tt.styles.ascii_thin_double)
		print(tableau)

	# Créer des graphiques des résultats
	elif option == "10":
		print(option + " : Créer graphiques des résultats")
		if test_lances == 1:
			# Pour nommer les figures créées
			date = time.strftime("%d_%m")
			hour = time.strftime("%H_%M_%S")
		
			# Graphique erreur L2 en fonction de dX
			# Extraire valeurs de L2 error
			L2error_index = resultats_titres.index('Erreur L2')
			ydata = [0]*len(table_resultats)
			for point in range(0,len(table_resultats),1):
				ydata[point] = table_resultats[point][L2error_index]

			# Créer valeurs de 1/dX
			# Obtenir les valeurs du refinement de embedding
			embedding_values = [0]*len(table_valeurs)
			for point in range(0,len(table_valeurs),1):
				embedding_values[point] = int(table_valeurs[point][col_embedding])

			xdata = [1]*len(ydata)
			for point in range(0,len(ydata),1):
				xdata[point] = pow(2,embedding_values[point]-1)

			# Conversion à des array numpy
			ydata = np.array(ydata, dtype=float)
			xdata = np.array(xdata, dtype=float)
			logxdata = xdata.tolist()
			logydata = ydata.tolist()

			for point in range(0,len(logxdata),1):
				logxdata[point] = math.log(logxdata[point],10)
				logydata[point] = math.log(logydata[point],10)

			logxdata = np.array(logxdata, dtype=float)
			logydata = np.array(logydata, dtype=float)



			# Création du modèle linéaire à partir des données en logarithme
			X_mean = logxdata.mean()
			Y_mean = logydata.mean()

			num = 0
			den = 0

			for point in range(len(logxdata)):
				num += (logxdata[point] - X_mean)*(logydata[point] - Y_mean)
				den += (logxdata[point] - X_mean)**2

			m = num / den
			b = Y_mean - m*X_mean

			param = [0]*2
			param[1] = m
			param[0] = 10**b

			# Courbe avec les moindres carrés custom
			ans = (param[0] * pow(xdata,param[1]))

			# Création du graphique
			fig, ax = plt.subplots()
			ax.loglog(xdata,ydata, 'ro', color='red', label='Données')
			ax.loglog(xdata,ans, '--', color='blue', label='Droite optimisée')
			pointx = np.mean(xdata)
			pointy = 0.75*np.mean(ydata)

			# Équation de la droite
			ax.text(pointx, pointy, f'{round(param[0],3)}*x^{round(param[1],5)}')

			ax.set(xlabel='1/dX', ylabel='Erreur L2', title='Erreur L2 en fonction du raffinement')
			   
			ax.grid(True, which='both', axis='both')
			ax.legend()

			graphname = 'Erreur L2'+" "+date+" "+hour+".png"
			fig.savefig(graphname)
			print("Graphique imprimé, fermer la fenêtre Matplotlib pour continuer d'utiliser le script.")
			print('Coefficient de corrélation R2: %.3f' % r2_score(ydata, ans))
			print("Mean squared error: %.3f" % mean_squared_error(ydata, ans))
			plt.show()



		else:
			print("Aucun test n'a été lancé!")
					   

	# Permet de quitter le programme
	elif option == '000':
		break
sys.exit
