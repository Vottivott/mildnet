source settings.cfg
source $1
echo $1
python -m trainer.task --job-dir local_job_dir \
--data-path=$MildNET_DATA_PATH \
--model-id=$model_id \
--loss=$loss \
--optimizer=$optimizer \
--train-csv=$train_csv \
--val-csv=$val_csv \
--train-epocs=$train_epocs \
--lr=$lr