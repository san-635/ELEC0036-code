# adapted from https://wfdb.io/mimic_wfdb_tutorials/tutorials.html
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import wfdb

# functions used in 8_extract_ppg.py

def get_records(database_name):
    subjects = wfdb.get_record_list(database_name)
    print(f'\nThe database contains waveform data from {len(subjects)} subjects')

    records = []
    for subject in subjects:
        studies = wfdb.get_record_list(f'{database_name}/{subject}')
        for study in studies:
            records.append(Path(f'{subject}{study}'))   # append waves\<dir>\<subject_id>\<record_name>\<record_name>

    """ Note:
    - intermediate directory / dir (e.g. 'p100')
    - subject identifier / subject_id (e.g. 'p10014354')
    - record identifier / record_name (e.g. '81739927')"""

    return records

def validate_segments(database_name, records, output_path, req_sig=['Pleth'], req_seg_duration=2*60):
    matching_recs = {'dir':[], 'subject_id': [], 'record_name':[], 'seg_name':[], 'length':[]} # dict to store relevant metadata of valid segments

    for record in tqdm(records, total=len(records), desc='\nExtracting valid segments across all records'):
        record_dir = database_name+'/'+str(record.parent).replace('\\', '/')
        record_name = record.name
        subject_id = int(str(record.parent.parent.name)[1:])
        print(f'\nsubject_id : {subject_id}')

        # Check if the record is valid (i.e., contains the required PPG signals)
        record_data = wfdb.rdheader(record_name, pn_dir=record_dir, rd_segments=True)   # read multi-segment header
        sigs_present = record_data.sig_name
        if not all(x in sigs_present for x in req_sig):
            print(f'Record {record_name} is missing PPG signals')
            continue    # skip remaining steps for invalid records

        # Extract valid segments of the valid records (note: some segments of a record may not contain 2mins long PPG signals)
        segments = record_data.seg_name
        gen = (segment for segment in segments if segment != '~')
        for segment in gen:
            segment_metadata = wfdb.rdheader(record_name=segment, pn_dir=record_dir)    # read segment header
            seg_length = segment_metadata.sig_len/(segment_metadata.fs) # segment length in seconds

            if seg_length < req_seg_duration:
                print(f'Segment {segment} of record {record_name} is too short at {seg_length/60:.1f} mins')
                continue    # skip remaining steps for invalid segments of a valid record

            seg_length_min = float(seg_length/60)
            sigs_present = segment_metadata.sig_name
            
            if all(x in sigs_present for x in req_sig):
                matching_recs['subject_id'].append(subject_id)
                matching_recs['dir'].append(record_dir)                
                matching_recs['record_name'].append(record_name)
                matching_recs['seg_name'].append(segment)
                matching_recs['length'].append(seg_length_min)

                print(f'Segment {segment} of record {record_name} met all requirements')
            else:
                print(f'Segment {segment} of record {record_name} is long enough, but missing PPG signals')   # also an invalid segment of a valid record
    
    matching_recs = pd.DataFrame(matching_recs)
    matching_recs.to_csv(os.path.join(output_path, 'valid_segments.csv'), index=False)
    return matching_recs

def extract_ppg(matching_recs, output_path):
    for i,_ in tqdm(enumerate(matching_recs['seg_name']), total=len(matching_recs['seg_name']), desc='Extracting PPG signals'):
        # get relevant metadata of this segment from matching_recs
        subject_id = matching_recs['subject_id'][i]     
        record_dir = matching_recs['dir'][i]
        record_name = matching_recs['record_name'][i]
        segment_name = matching_recs['seg_name'][i]

        # create subject's sub-directory and output_file paths
        output_dir = os.path.join(output_path, str(subject_id))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, f'{record_name}_{segment_name}.csv')

        # extract PPG signals from this segment
        segment_data = wfdb.rdrecord(record_name = segment_name, pn_dir = record_dir)
        sig_no = segment_data.sig_name.index('Pleth')
        ppg = segment_data.p_signal[:, sig_no]
        ppg = ppg.tolist() # convert np array to list before saving to csv
        fs = float(segment_data.fs)

        # save the PPG signals and relevant metadata to the output_file
        ppg_df = pd.DataFrame([[subject_id, record_dir, record_name, segment_name, ppg, fs]], columns=['subject_id', 'record_dir', 'record_name', 'segment_name', 'ppg', 'sampling_frequency'])
        ppg_df.to_csv(output_file, index=False)