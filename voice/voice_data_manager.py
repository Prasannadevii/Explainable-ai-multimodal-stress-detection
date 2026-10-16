import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import shutil
from datetime import datetime

class VoiceDataManager:
    """
    Manage, organize, and analyze collected voice stress data
    """
    
    def __init__(self, data_dir="voice_stress_data"):
        self.data_dir = Path(data_dir)
    
    def get_dataset_summary(self):
        """
        Get summary statistics of collected data
        """
        summary = {
            'total_participants': 0,
            'total_sessions': 0,
            'total_recordings': 0,
            'by_level': {1: 0, 2: 0, 3: 0},
            'participants': {}
        }
        
        if not self.data_dir.exists():
            print("Data directory not found!")
            return summary
        
        # Iterate through all participants
        for participant_dir in self.data_dir.iterdir():
            if participant_dir.is_dir():
                summary['total_participants'] += 1
                participant_id = participant_dir.name
                summary['participants'][participant_id] = {
                    'sessions': 0,
                    'recordings': 0,
                    'by_level': {1: 0, 2: 0, 3: 0}
                }
                
                # Iterate through sessions
                for session_dir in participant_dir.iterdir():
                    if session_dir.is_dir():
                        summary['total_sessions'] += 1
                        summary['participants'][participant_id]['sessions'] += 1
                        
                        # Count recordings
                        wav_files = list(session_dir.glob("*.wav"))
                        summary['participants'][participant_id]['recordings'] += len(wav_files)
                        summary['total_recordings'] += len(wav_files)
                        
                        # Count by level
                        for wav_file in wav_files:
                            json_file = wav_file.with_suffix('.json')
                            if json_file.exists():
                                with open(json_file, 'r') as f:
                                    metadata = json.load(f)
                                    level = metadata.get('level', 0)
                                    summary['by_level'][level] = summary['by_level'].get(level, 0) + 1
                                    summary['participants'][participant_id]['by_level'][level] += 1
        
        return summary
    
    def print_summary(self):
        """
        Print dataset summary to console
        """
        summary = self.get_dataset_summary()
        
        print("\n" + "="*60)
        print("VOICE DATA COLLECTION SUMMARY")
        print("="*60)
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Participants: {summary['total_participants']}")
        print(f"  Total Sessions: {summary['total_sessions']}")
        print(f"  Total Recordings: {summary['total_recordings']}")
        
        print(f"\n📈 Recordings by Stress Level:")
        for level in [1, 2, 3]:
            count = summary['by_level'].get(level, 0)
            level_names = {1: "Normal", 2: "Medium", 3: "High"}
            print(f"  Level {level} ({level_names[level]}): {count} recordings")
        
        print(f"\n👥 Participant Details:")
        for participant_id, data in summary['participants'].items():
            print(f"\n  {participant_id}:")
            print(f"    Sessions: {data['sessions']}")
            print(f"    Total Recordings: {data['recordings']}")
            print(f"    By Level - Normal: {data['by_level'].get(1, 0)}, Medium: {data['by_level'].get(2, 0)}, High: {data['by_level'].get(3, 0)}")
    
    def export_to_csv(self, output_file="voice_data_summary.csv"):
        """
        Export data summary to CSV
        """
        summary = self.get_dataset_summary()
        
        rows = []
        for participant_id, data in summary['participants'].items():
            rows.append({
                'Participant_ID': participant_id,
                'Sessions': data['sessions'],
                'Total_Recordings': data['recordings'],
                'Level_1_Normal': data['by_level'].get(1, 0),
                'Level_2_Medium': data['by_level'].get(2, 0),
                'Level_3_High': data['by_level'].get(3, 0)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"✅ Summary exported to {output_file}")
        return df
    
    def validate_data(self):
        """
        Validate data integrity - check for orphaned files, missing metadata, etc.
        """
        issues = {
            'orphaned_wav': [],
            'orphaned_json': [],
            'missing_metadata': [],
            'corrupted_metadata': []
        }
        
        print("\n🔍 Validating data integrity...")
        
        for wav_file in self.data_dir.glob("*/*/*.wav"):
            json_file = wav_file.with_suffix('.json')
            
            if not json_file.exists():
                issues['missing_metadata'].append(str(wav_file))
            else:
                try:
                    with open(json_file, 'r') as f:
                        json.load(f)
                except json.JSONDecodeError:
                    issues['corrupted_metadata'].append(str(json_file))
        
        # Check for orphaned JSON files
        for json_file in self.data_dir.glob("*/*/*.json"):
            wav_file = json_file.with_suffix('.wav')
            if not wav_file.exists():
                issues['orphaned_json'].append(str(json_file))
        
        # Print results
        print("\n📋 Validation Results:")
        print(f"  Missing Metadata: {len(issues['missing_metadata'])}")
        if issues['missing_metadata']:
            for file in issues['missing_metadata'][:5]:
                print(f"    - {file}")
            if len(issues['missing_metadata']) > 5:
                print(f"    ... and {len(issues['missing_metadata']) - 5} more")
        
        print(f"  Corrupted Metadata: {len(issues['corrupted_metadata'])}")
        print(f"  Orphaned JSON files: {len(issues['orphaned_json'])}")
        
        if all(len(v) == 0 for v in issues.values()):
            print("\n✅ All data is valid!")
        else:
            print("\n⚠️  Some issues found. Review above.")
        
        return issues
    
    def get_participant_data(self, participant_id):
        """
        Get detailed info about a specific participant
        """
        participant_dir = self.data_dir / participant_id
        
        if not participant_dir.exists():
            print(f"Participant {participant_id} not found!")
            return None
        
        data = {
            'participant_id': participant_id,
            'sessions': []
        }
        
        for session_dir in participant_dir.iterdir():
            if session_dir.is_dir():
                session = {
                    'session_id': session_dir.name,
                    'recordings': []
                }
                
                for wav_file in sorted(session_dir.glob("*.wav")):
                    json_file = wav_file.with_suffix('.json')
                    rec_info = {'file': wav_file.name}
                    
                    if json_file.exists():
                        with open(json_file, 'r') as f:
                            rec_info.update(json.load(f))
                    
                    session['recordings'].append(rec_info)
                
                data['sessions'].append(session)
        
        return data
    
    def delete_recording(self, participant_id, session_id, filename):
        """
        Delete a specific recording (with confirmation)
        """
        wav_file = self.data_dir / participant_id / session_id / filename
        json_file = wav_file.with_suffix('.json')
        
        if not wav_file.exists():
            print(f"File not found: {filename}")
            return False
        
        # Create backup before deletion
        backup_dir = self.data_dir / "_backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_wav = backup_dir / f"{participant_id}_{session_id}_{wav_file.name}"
        backup_json = backup_dir / f"{participant_id}_{session_id}_{json_file.name}"
        
        shutil.copy2(wav_file, backup_wav)
        if json_file.exists():
            shutil.copy2(json_file, backup_json)
        
        # Delete original
        wav_file.unlink()
        if json_file.exists():
            json_file.unlink()
        
        print(f"✅ Deleted {filename} (backed up in _backups/)")
        return True
    
    def get_statistics(self):
        """
        Get detailed statistics about collected data
        """
        summary = self.get_dataset_summary()
        stats = {
            'avg_sessions_per_participant': 0,
            'avg_recordings_per_session': 0,
            'avg_recordings_per_participant': 0,
            'recordings_distribution': {1: 0, 2: 0, 3: 0}
        }
        
        if summary['total_participants'] > 0:
            stats['avg_sessions_per_participant'] = summary['total_sessions'] / summary['total_participants']
        
        if summary['total_sessions'] > 0:
            stats['avg_recordings_per_session'] = summary['total_recordings'] / summary['total_sessions']
        
        if summary['total_participants'] > 0:
            stats['avg_recordings_per_participant'] = summary['total_recordings'] / summary['total_participants']
        
        stats['recordings_distribution'] = summary['by_level']
        
        return stats


if __name__ == "__main__":
    from pathlib import Path as _Path
    _PROJECT_ROOT = _Path(__file__).resolve().parent.parent
    _VOICE_DATA_DIR = _PROJECT_ROOT / "data" / "voice" / "voice_stress_data"
    _SUMMARY_CSV = _PROJECT_ROOT / "data" / "voice" / "voice_data_summary.csv"

    manager = VoiceDataManager(data_dir=str(_VOICE_DATA_DIR))
    
    # Print summary
    manager.print_summary()
    
    # Validate data
    manager.validate_data()
    
    # Export summary
    manager.export_to_csv(output_file=str(_SUMMARY_CSV))
    
    print("\n✅ Data management complete!")
