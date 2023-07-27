mkdir raw_data

# Adult Census
wget -c https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget -c https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
wget -c https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
mkdir raw_data/adult
mv adult.* raw_data/adult/

# Bank Marketting
wget -c https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
mkdir raw_data/bank_marketing
mv bank-additional.zip raw_data/bank_marketing
cd raw_data/bank_marketing
unzip bank-additional.zip
mv bank-additional/* .
rm -r -f bank-additional/

