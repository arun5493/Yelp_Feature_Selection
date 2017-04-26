import json
import csv
import pandas as pd
import numpy as np
import unicodedata
import ast
from tqdm import tqdm
import operator

def parse_data(source1,output_path):
    
    # extract fields from business json data set #
    # setup an array for storing each json entry
	business_data = []

    # open the business source file
	with open(source1) as f:	
		# for each line in the json file
		for line in f:
			# store the line in the array for manipulation
			business_data.append(json.loads(line))
	# close the reader
	f.close()

	#print business_data[1].keys()

	# allCategoriesList = []
	# for entry in tqdm(range(0, len(business_data))):
	# 	categoryList = [] 
	# 	categoryList.append(business_data[entry]['categories'])
	# 	#print categoryList[0]
	# 	if categoryList[0] is not None:
	# 		for each in categoryList[0]:
	# 			allCategoriesList.append(each)

	# categoriesCount = {}
	# for each in allCategoriesList:
	# 	if each not in categoriesCount.keys():
	# 		categoriesCount[each] = 0
	# 	categoriesCount[each]+=1
	# sorted_x = sorted(categoriesCount.items(), key=operator.itemgetter(1))
	#print sorted_x


	print('processing data in the business dataset...')
    # for every entry in the business data array
	headerRow = ['business_id','address','city','state','postal_code','name','review_count','stars','type','BikeParking','BusinessAcceptsBitcoin',
				'BusinessAcceptsCreditCards','WheelchairAccessible','Caters','GoodForKids','HasTV','OutdoorSeating','RestaurantsDelivery',
				'RestaurantsGoodForGroups','RestaurantsReservations','RestaurantsTakeOut','RestaurantsTableService',
				'BusinessParking_garage','BusinessParking_street','BusinessParking_validated','BusinessParking_lot','BusinessParking_valet',
				'RestaurantsPriceRange',
				'Ambience_romantic','Ambience_intimate','Ambience_classy','Ambience_hipster','Ambience_divey','Ambience_touristy','Ambience_trendy','Ambience_upscale','Ambience_casual',
				'Alcohol','WiFi',
				'Music_DJ','Music_Background','Music_Karaoke','Music_Live','Music_Video','Music_Jukebox',
				'Categories_Restaurant','Categories_Food','Categories_NightLife','Categories_Bars','Categories_AmericanTraditional','Categories_FastFood','Categories_Pizza','Categories_Sandwiches',
				'Categories_Coffee&Tea','Categories_Italian','Categories_Burgers','Categories_Mexican','Categories_AmericanNew','Categories_Chinese','Categories_Breakfast&Brunch',
				'Categories_SpecialityFoods','Categories_Cafes','Categories_Hotels','Categories_Desserts','Categories_Japanese','Categories_IceCreams','Categories_ChickenWings',
				'Categories_SeaFood','Categories_Sushi Bars','Categories_SportsBars','Categories_Beer','Categories_Wine','Categories_Delis','Categories_Asian','Categories_Salad',
				'Categories_Med','Categories_Barbeque','Categories_Indian','Categories_SteakHouses','Categories_Thai','Categories_Diners','Categories_French','Categories_Greek',
				'Categories_Vegetarian','Categories_Buffet','Categories_GlutenFree','Categories_Soup','Categories_Vegan']

	attributeSubset = ['BikeParking','BusinessAcceptsBitcoin','BusinessAcceptsCreditCards','WheelchairAccessible','Caters','GoodForKids','HasTV',
						'OutdoorSeating','RestaurantsDelivery','RestaurantsGoodForGroups','RestaurantsReservations','RestaurantsTakeOut','RestaurantsTableService']

	businessParkingList = ['garage','street','validated','lot','valet']			
	ambienceList = ['romantic','intimate','classy','hipster','divey','touristy','trendy','upscale','casual']
	musicList = ['dj','background_music','karaoke','live','video','jukebox']
	allCategoriesList = ['Vegan','Soup','Gluten-Free','Buffets','Vegetarian','Greek','French','Diners','Thai','Indian','Steakhouses','Barbeque','Mediterranean','Salad','Asian Fusion',
						'Delis','Wine & Spirits','Beer','Sports Bars','Sushi Bars','Seafood','Chicken Wings','Ice Cream & Frozen Yogurt','Japanese','Desserts','Hotels','Cafes',
						'Specialty Food','Breakfast & Brunch','Chinese','American (New)','Mexican','Burgers','Italian','Coffee & Tea','Sandwiches','Pizza','Fast Food',
						'American (Traditional)','Bars','Nightlife','Food','Restaurants']
	#print len(allCategoriesList)
	

	modified_data = []
	for entry in tqdm(range(0, len(business_data))):
	#for entry in tqdm(range(0, 100)):    	
		row = []    
		row.append(business_data[entry]['business_id'])
		row.append(business_data[entry]['address'])
		row.append(business_data[entry]['city'])
		row.append(business_data[entry]['state'])
		row.append(business_data[entry]['postal_code'])
		row.append(business_data[entry]['name'])
		row.append(business_data[entry]['review_count'])
		row.append(business_data[entry]['stars'])
		row.append(business_data[entry]['type'])

		# if business_data[entry]['business_id'] == 'CGliHrLYH8ABT3k19kLF3w':	#This Data is not added to CSV. Need to check why
		# 	print business_data[entry]

		if(business_data[entry]['attributes'] is not None):
			attributesList = business_data[entry]['attributes']
			for each in attributeSubset:
				if(each+': True' in attributesList):
					row.append(1)
				else:
					row.append(0)			

			businessParkingFound = False
			RestaurantsPriceRangeFound = False
			ambienceFound = False
			musicFound = False
			for element in attributesList:
				if('BusinessParking' in element):
					businessParkingFound = True
					# To convert this string into a dictionary we use ast.literal_eval
					parkingDict = ast.literal_eval(element[len('BusinessParking: '):])

					for each in businessParkingList:
						if each in parkingDict.keys() and parkingDict[each]:
							row.append(1)
						else:
							row.append(0)
			# If BusinessParking attribute is not present in the file, then we assume there is no business parking	
			if(businessParkingFound == False):
				row.extend([0]*5)
				
			for element in attributesList:
				if('RestaurantsPriceRange2' in element):
					RestaurantsPriceRangeFound = True
					row.append(element[len('RestaurantsPriceRange2 '):])		# Watch out for its type.
			if(RestaurantsPriceRangeFound == False):
				row.append(0)

			for element in attributesList:		
				if('Ambience' in element):
					ambienceFound = True
					# To convert this string into a dictionary we use ast.literal_eval
					ambienceDict = ast.literal_eval(element[len('Ambience: '):])
					#print ambienceDict
					for each in ambienceList:
						if each in ambienceDict.keys() and ambienceDict[each]:
							row.append(1)
						else:
							row.append(0)

			if(ambienceFound == False):
				row.extend([0]*9)
			

			if('Alcohol: full_bar' in attributesList):
				row.append(2)
			elif('Alcohol: beer_and_wine' in attributesList):
				row.append(1)
			else:
				row.append(0)
			
			
			if('WiFi: free' in attributesList):
				row.append(2)
			elif ('WiFi: paid' in attributesList):
				row.append(1)
			else:
				row.append(0)

			for element in attributesList:		
				if('Music: ' in element):
					musicFound = True
					# To convert this string into a dictionary we use ast.literal_eval
					musicDict = ast.literal_eval(element[len('Music: '):])
					#print musicDict
					for each in musicList:
						if each in musicDict.keys() and musicDict[each]:
							row.append(1)
						else:
							row.append(0)

			if(musicFound == False):
				row.extend([0]*6)
				
		else:
			row.extend([0]*35)		# If no attribute is  not found, then appending 0 for all the attributes


		categoryFound = False
		if(business_data[entry]['categories'] is not None):
			currCategoriesList = business_data[entry]['categories']
			for each in currCategoriesList:
				for element in allCategoriesList:
					if each == element:
						categoryFound = True
						break
	
			if(categoryFound == False):
				continue		# If the categories is not found, then not adding that row  in the data			
			else:
				for each in allCategoriesList:
					if each in currCategoriesList:
						row.append(1)
					else:
						row.append(0)

		else:
			continue			# Not including those rows without any categories				



		modified_data.append(row)
	
	
	#write to csv file
	print len(modified_data)
	with open(output_path, 'w') as out:
		writer = csv.writer(out)
		# write the csv headers
		writer.writerow(headerRow)
		# for each entry in the row array
		print('writing contents to csv...')
		for entry in tqdm(range(0, len(modified_data))):
			try:
				# write the row to the csv
				writer.writerow(modified_data[entry])
			# if there is an error, continue to the next row
			except UnicodeEncodeError:
				#print "There is an error now"
				continue
	out.close()
	

parse_data('yelp_academic_dataset_business.json','test.csv')