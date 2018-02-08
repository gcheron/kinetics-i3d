

job_list="job_list"

n=0
#for i in /sequoia/data1/gcheron/code/torch/lstm_time_detection/dataset/splitlists/sub/all_vidlist_sub*
for i in /sequoia/data1/gcheron/code/torch/lstm_time_detection/dataset/splitlistsDALY/sub/all_vidlist_sub*
do
n=$(($n+1))

JOBNAME=rgbconv$n

{
echo "#$ -l mem_req=5G"
echo "#$ -l h_vmem=400G"
echo "#$ -j y"
echo "#$ -o /sequoia/data1/gcheron/code/tensorflow/kinetics-i3d/preproc_jobs/logs"
echo "#$ -N $JOBNAME"
echo "#$ -q all.q"

echo "echo \$HOSTNAME"
echo "export PATH="/sequoia/data1/gcheron/lib/anaconda2/bin:/sequoia/data3/gcheron/torch/torch_install_bigmem/install/bin:/cm/shared/apps/sge/2011.11p1/bin/linux-x64:/cm/shared/apps/gcc/4.8.1/bin:/usr/lib64/qt-3.3/bin:/usr/local/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/sbin:/sbin:/usr/sbin:/opt/dell/srvadmin/bin:/meleze/data0/local/bin:/usr/local/cuda-7.5/bin::/sequoia/data3/gcheron/torch/distr_bigmem/git-2.9.3:/home/gcheron/bin"
echo "export LD_LIBRARY_PATH="/sequoia/data3/gcheron/torch/torch_install_bigmem/install/lib:/usr/local/cudnn/5.0/lib64:/sequoia/data2/gpunodes_shared_libs/cudnn/5.0/lib64:/cm/shared/apps/sge/2011.11p1/lib/linux-x64:/cm/shared/apps/gcc/4.8.1/lib:/cm/shared/apps/gcc/4.8.1/lib64:/usr/local/cuda-7.5/lib64:"

echo "cd /sequoia/data1/gcheron/code/tensorflow/kinetics-i3d"
echo "python preprocess_rgbopf.py $i rgb"
echo "echo JOB DONE"
} > $job_list/$JOBNAME.pbs

done
