"""
Data Loader Module
Time of Flight 실험 데이터 파일을 로드하고 검증하는 모듈
"""
import json
import xarray as xr
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime

class TOFDataLoader:
    """실험 데이터 로드 및 검증 클래스"""
    
    def __init__(self):
        """데이터 로더 초기화"""
        # 필수 파일 목록
        self.required_files = {
            'metadata': 'metadata.json',
            'ds_raw': 'ds_raw.nc',
            'ds_fit': 'ds_fit.nc',
            'qubit_info': 'qubit_info.json'
        }
        
        # 선택적 파일 목록
        self.optional_files = {
            'analysis_results': 'analysis_results.json',
            'experiment_config': 'experiment_config.json',
            'notes': 'notes.txt'
        }
        
        # 지원하는 실험 타입
        self.supported_experiments = {
            'time_of_flight',
            'resonator_spectroscopy',
            'qubit_spectroscopy',
            'ramsey',
            'rabi_amplitude',
            'rabi_power',
            't1',
            't2',
            't2_echo'
        }
    
    def load_experiment(self, experiment_dir: Path) -> Optional[Dict]:
        """실험 폴더에서 데이터 로드
        
        Parameters
        ----------
        experiment_dir : Path
            실험 데이터가 있는 디렉토리
            
        Returns
        -------
        Dict or None
            로드된 실험 데이터 또는 실패 시 None
        """
        try:
            # 디렉토리 검증
            if not experiment_dir.is_dir():
                raise ValueError(f"Not a directory: {experiment_dir}")
            
            # 필수 파일 확인
            self._validate_required_files(experiment_dir)
            
            # 메타데이터 로드
            metadata = self._load_metadata(experiment_dir)
            
            # 실험 타입 검증
            exp_type = metadata.get('experiment_type')
            if exp_type not in self.supported_experiments:
                print(f"⚠️  Warning: Unknown experiment type '{exp_type}'")
            
            # xarray 데이터셋 로드
            ds_raw = self._load_dataset(experiment_dir / self.required_files['ds_raw'])
            ds_fit = self._load_dataset(experiment_dir / self.required_files['ds_fit'])
            
            # Qubit 정보 로드
            qubit_info = self._load_json(experiment_dir / self.required_files['qubit_info'])
            
            # 선택적 파일 로드
            optional_data = self._load_optional_files(experiment_dir)
            
            # 데이터 통합
            experiment_data = {
                'type': exp_type,
                'timestamp': metadata.get('timestamp_full', metadata.get('timestamp')),
                'ds_raw': ds_raw,
                'ds_fit': ds_fit,
                'qubit_info': qubit_info,
                'metadata': metadata,
                **optional_data  # 선택적 데이터 추가
            }
            
            # 데이터 검증
            self._validate_experiment_data(experiment_data)
            
            return experiment_data
            
        except Exception as e:
            print(f"Error loading experiment from {experiment_dir}: {e}")
            return None
    
    def _validate_required_files(self, experiment_dir: Path):
        """필수 파일 존재 확인"""
        missing_files = []
        
        for file_type, filename in self.required_files.items():
            file_path = experiment_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {experiment_dir.name}: {', '.join(missing_files)}"
            )
    
    def _load_metadata(self, experiment_dir: Path) -> Dict:
        """메타데이터 로드 및 검증"""
        metadata_path = experiment_dir / self.required_files['metadata']
        metadata = self._load_json(metadata_path)
        
        # 필수 메타데이터 필드 확인
        required_fields = ['experiment_id', 'experiment_type', 'timestamp']
        missing_fields = [f for f in required_fields if f not in metadata]
        
        if missing_fields:
            raise ValueError(f"Missing required metadata fields: {', '.join(missing_fields)}")
        
        # 타임스탬프 파싱 (필요시)
        if 'timestamp_full' not in metadata and 'timestamp' in metadata:
            try:
                # 타임스탬프 형식 추정 및 파싱
                timestamp_str = metadata['timestamp']
                if len(timestamp_str) == 15:  # YYYYMMDD_HHMMSS 형식
                    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    metadata['timestamp_full'] = dt.isoformat()
            except:
                pass
        
        return metadata
    
    def _load_dataset(self, file_path: Path) -> xr.Dataset:
        """xarray 데이터셋 로드"""
        try:
            ds = xr.open_dataset(file_path)
            
            # 데이터셋 기본 검증
            if len(ds.data_vars) == 0:
                raise ValueError(f"Empty dataset: {file_path.name}")
            
            return ds
            
        except Exception as e:
            raise ValueError(f"Failed to load dataset {file_path.name}: {e}")
    
    def _load_json(self, file_path: Path) -> Dict:
        """JSON 파일 로드"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path.name}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load {file_path.name}: {e}")
    
    def _load_optional_files(self, experiment_dir: Path) -> Dict:
        """선택적 파일 로드"""
        optional_data = {}
        
        for file_type, filename in self.optional_files.items():
            file_path = experiment_dir / filename
            if file_path.exists():
                try:
                    if filename.endswith('.json'):
                        optional_data[file_type] = self._load_json(file_path)
                    elif filename.endswith('.txt'):
                        with open(file_path, 'r') as f:
                            optional_data[file_type] = f.read()
                    print(f"   📄 Loaded optional: {filename}")
                except Exception as e:
                    print(f"   ⚠️  Failed to load optional {filename}: {e}")
        
        return optional_data
    
    def _validate_experiment_data(self, experiment_data: Dict):
        """로드된 실험 데이터 검증"""
        # 데이터셋 차원 일치 확인
        ds_raw = experiment_data['ds_raw']
        ds_fit = experiment_data['ds_fit']
        qubit_info = experiment_data['qubit_info']
        
        # Qubit 차원 확인
        if 'qubit' in ds_raw.dims:
            n_qubits_raw = len(ds_raw.qubit)
            n_qubits_info = len(qubit_info.get('grid_locations', []))
            
            if n_qubits_raw != n_qubits_info:
                print(f"⚠️  Warning: Qubit count mismatch - "
                      f"Dataset: {n_qubits_raw}, Info: {n_qubits_info}")
        
        # 실험 타입별 추가 검증
        exp_type = experiment_data['type']
        self._validate_experiment_specific(exp_type, experiment_data)
    
    def _validate_experiment_specific(self, exp_type: str, experiment_data: Dict):
        """실험 타입별 특수 검증"""
        ds_raw = experiment_data['ds_raw']
        ds_fit = experiment_data['ds_fit']
        
        if exp_type == 'time_of_flight':
            required_vars = ['adcI', 'adcQ']
            missing_vars = [v for v in required_vars if v not in ds_raw.data_vars]
            if missing_vars:
                print(f"⚠️  Warning: Missing expected variables for TOF: {missing_vars}")
            
            if 'delay' not in ds_fit.data_vars:
                print("⚠️  Warning: 'delay' not found in fit results")
        
    def get_experiment_summary(self, experiment_data: Dict) -> Dict:
        """실험 데이터 요약 정보 생성"""
        metadata = experiment_data['metadata']
        ds_raw = experiment_data['ds_raw']
        qubit_info = experiment_data['qubit_info']
        
        summary = {
            'experiment_id': metadata['experiment_id'],
            'experiment_type': metadata['experiment_type'],
            'timestamp': metadata.get('timestamp_full', metadata['timestamp']),
            'n_qubits': len(qubit_info.get('grid_locations', [])),
            'data_shape': dict(ds_raw.dims),
            'data_vars': list(ds_raw.data_vars),
            'coordinates': list(ds_raw.coords),
            'optional_data': list(k for k in experiment_data.keys() 
                               if k not in ['type', 'timestamp', 'ds_raw', 
                                          'ds_fit', 'qubit_info', 'metadata'])
        }
        
        return summary
    
    def save_experiment_summary(self, experiment_dir: Path, summary: Dict):
        """실험 요약 정보 저장 (캐싱용)"""
        summary_path = experiment_dir / 'summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"   💾 Saved experiment summary")
        except Exception as e:
            print(f"   ⚠️  Failed to save summary: {e}")
    
    @staticmethod
    def validate_dataset_compatibility(ds1: xr.Dataset, ds2: xr.Dataset) -> bool:
        """두 데이터셋의 호환성 확인 (비교 분석용)"""
        # 차원 확인
        if ds1.dims.keys() != ds2.dims.keys():
            return False
        
        # 좌표 확인
        for coord in ds1.coords:
            if coord not in ds2.coords:
                return False
            if not ds1[coord].equals(ds2[coord]):
                return False
        
        return True