"""
Universal Data Loader Module
모든 실험 타입을 지원하는 범용 데이터 로더
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

from .tof_data_loader import TOFDataLoader
from .res_spec_data_loader import ResonatorSpecDataLoader

class UniversalDataLoader:
    """모든 실험 타입을 처리하는 범용 로더"""
    
    def __init__(self):
        """범용 로더 초기화"""
        # 실험 타입별 로더 레지스트리
        self.loaders = {
            'time_of_flight': TOFDataLoader(),
            'resonator_spectroscopy': ResonatorSpecDataLoader(),
            # 추후 다른 실험 타입 로더 추가
            # 'qubit_spectroscopy': QubitSpecDataLoader(),
            # 'ramsey': RamseyDataLoader(),
            # 'rabi_amplitude': RabiDataLoader(),
        }
        
        # 지원하는 실험 타입
        self.supported_experiments = set(self.loaders.keys())
        
        # 기본 로더 (실험 타입을 확인하기 위한 용도)
        self.default_loader = TOFDataLoader()
    
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
            # 먼저 메타데이터만 로드하여 실험 타입 확인
            metadata_path = experiment_dir / 'metadata.json'
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            exp_type = metadata.get('experiment_type')
            if not exp_type:
                raise ValueError("Experiment type not found in metadata")
            
            print(f"🔍 Detected experiment type: {exp_type}")
            
            # 해당 실험 타입의 로더 선택
            if exp_type in self.loaders:
                loader = self.loaders[exp_type]
                print(f"✓ Using specialized loader for {exp_type}")
            else:
                # 지원하지 않는 실험 타입은 기본 로더 사용
                print(f"⚠️  No specialized loader for {exp_type}, using default loader")
                loader = self.default_loader
            
            # 선택된 로더로 데이터 로드
            return loader.load_experiment(experiment_dir)
            
        except Exception as e:
            print(f"❌ Error loading experiment from {experiment_dir}: {e}")
            return None
    
    def get_experiment_type(self, experiment_dir: Path) -> Optional[str]:
        """실험 디렉토리에서 실험 타입만 빠르게 확인
        
        Parameters
        ----------
        experiment_dir : Path
            실험 데이터가 있는 디렉토리
            
        Returns
        -------
        str or None
            실험 타입 또는 실패 시 None
        """
        try:
            metadata_path = experiment_dir / 'metadata.json'
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata.get('experiment_type')
            
        except Exception:
            return None
    
    def is_experiment_supported(self, exp_type: str) -> bool:
        """실험 타입이 전용 로더를 가지고 있는지 확인
        
        Parameters
        ----------
        exp_type : str
            실험 타입
            
        Returns
        -------
        bool
            전용 로더가 있으면 True
        """
        return exp_type in self.loaders
    
    def get_supported_experiments(self) -> set:
        """지원하는 실험 타입 목록 반환
        
        Returns
        -------
        set
            지원하는 실험 타입들의 집합
        """
        return self.supported_experiments.copy()
    
    def register_loader(self, exp_type: str, loader: Any):
        """새로운 실험 타입 로더 등록
        
        Parameters
        ----------
        exp_type : str
            실험 타입
        loader : Any
            해당 실험 타입의 로더 인스턴스
        """
        self.loaders[exp_type] = loader
        self.supported_experiments.add(exp_type)
        print(f"✓ Registered loader for {exp_type}")
    
    def get_loader_for_type(self, exp_type: str) -> Any:
        """특정 실험 타입의 로더 반환
        
        Parameters
        ----------
        exp_type : str
            실험 타입
            
        Returns
        -------
        Any
            해당 실험 타입의 로더 또는 기본 로더
        """
        return self.loaders.get(exp_type, self.default_loader)