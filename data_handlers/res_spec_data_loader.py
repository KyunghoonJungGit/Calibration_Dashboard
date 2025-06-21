"""
Resonator Spectroscopy Data Loader Module
Load and validate Resonator Spectroscopy experiment data for Plotly Dash dashboard
공진기 분광 실험 데이터를 로드하고 검증하는 모듈
"""
import json
import xarray as xr
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np


class ResonatorSpecDataLoader:
    """Resonator Spectroscopy 실험 데이터 로드 및 검증 클래스"""
    
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
        
        # Resonator Spectroscopy 필수 데이터 변수
        self.required_data_vars = {
            'ds_raw': ['phase', 'IQ_abs'],
            'ds_fit': ['amplitude', 'position', 'width', 'base_line']
        }
        
        # 필수 좌표
        self.required_coords = ['full_freq', 'detuning', 'qubit']
    
    def load_experiment(self, experiment_dir: Path) -> Optional[Dict]:
        """
        Resonator Spectroscopy 실험 데이터 로드
        
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
            
            print(f"\n=== Loading Resonator Spectroscopy data from {experiment_dir.name} ===")
            
            # 필수 파일 확인
            self._validate_required_files(experiment_dir)
            
            # 메타데이터 로드
            metadata = self._load_metadata(experiment_dir)
            
            # 실험 타입 검증
            exp_type = metadata.get('experiment_type')
            if exp_type != 'resonator_spectroscopy':
                print(f"⚠️  Warning: Expected 'resonator_spectroscopy' but got '{exp_type}'")
            
            # xarray 데이터셋 로드
            ds_raw = self._load_and_validate_dataset(
                experiment_dir / self.required_files['ds_raw'], 
                'ds_raw'
            )
            ds_fit = self._load_and_validate_dataset(
                experiment_dir / self.required_files['ds_fit'], 
                'ds_fit'
            )
            
            # Qubit 정보 로드
            qubit_info = self._load_json(experiment_dir / self.required_files['qubit_info'])
            
            # 선택적 파일 로드
            optional_data = self._load_optional_files(experiment_dir)
            
            # 데이터 통합
            experiment_data = {
                'type': exp_type,
                'experiment_id': metadata.get('experiment_id'),
                'timestamp': metadata.get('timestamp_full', metadata.get('timestamp')),
                'ds_raw': ds_raw,
                'ds_fit': ds_fit,
                'qubit_info': qubit_info,
                'metadata': metadata,
                **optional_data
            }
            
            # 데이터 일관성 검증
            self._validate_data_consistency(experiment_data)
            
            # 추가 분석 정보 계산 (필요시)
            experiment_data['computed_info'] = self._compute_additional_info(experiment_data)
            
            # 성공 메시지 출력
            freq_values = ds_raw.full_freq.values
            # 주파수가 2차원인 경우 (qubit별로 다른 주파수)
            if freq_values.ndim == 2:
                print(f"✓ Successfully loaded Resonator Spectroscopy data")
                print(f"  - {len(qubit_info['grid_locations'])} qubits")
                print(f"  - {freq_values.shape[1]} frequency points per qubit")
                print(f"  - Frequency range (overall): {freq_values.min()/1e9:.3f} - {freq_values.max()/1e9:.3f} GHz")
            else:
                # 1차원인 경우 (모든 큐빗이 동일한 주파수)
                print(f"✓ Successfully loaded Resonator Spectroscopy data")
                print(f"  - {len(qubit_info['grid_locations'])} qubits")
                print(f"  - {len(freq_values)} frequency points")
                print(f"  - Frequency range: {freq_values.min()/1e9:.3f} - {freq_values.max()/1e9:.3f} GHz")
            
            return experiment_data
            
        except Exception as e:
            print(f"❌ Error loading experiment from {experiment_dir}: {e}")
            import traceback
            traceback.print_exc()
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
                f"Missing required files: {', '.join(missing_files)}"
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
        
        return metadata
    
    def _load_and_validate_dataset(self, file_path: Path, dataset_type: str) -> xr.Dataset:
        """xarray 데이터셋 로드 및 검증"""
        try:
            ds = xr.open_dataset(file_path)
            
            # 기본 검증
            if len(ds.data_vars) == 0:
                raise ValueError(f"Empty dataset: {file_path.name}")
            
            # Resonator Spectroscopy 특화 검증
            if dataset_type == 'ds_raw':
                # 필수 좌표 확인
                missing_coords = [c for c in self.required_coords if c not in ds.coords]
                if missing_coords:
                    raise ValueError(f"Missing coordinates in {file_path.name}: {missing_coords}")
                
                # 필수 데이터 변수 확인
                missing_vars = [v for v in self.required_data_vars['ds_raw'] if v not in ds.data_vars]
                if missing_vars:
                    print(f"⚠️  Warning: Missing data variables in {file_path.name}: {missing_vars}")
            
            elif dataset_type == 'ds_fit':
                # 피팅 파라미터 확인
                expected_vars = self.required_data_vars['ds_fit']
                present_vars = [v for v in expected_vars if v in ds.data_vars]
                if len(present_vars) < 3:  # 최소 3개 이상의 피팅 파라미터 필요
                    print(f"⚠️  Warning: Only {len(present_vars)} fit parameters found")
            
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
                    print(f"  📄 Loaded optional: {filename}")
                except Exception as e:
                    print(f"  ⚠️  Failed to load optional {filename}: {e}")
        
        return optional_data
    
    def _validate_data_consistency(self, experiment_data: Dict):
        """데이터 일관성 검증"""
        ds_raw = experiment_data['ds_raw']
        ds_fit = experiment_data['ds_fit']
        qubit_info = experiment_data['qubit_info']
        
        # Qubit 수 일치 확인
        n_qubits_raw = len(ds_raw.qubit)
        n_qubits_fit = len(ds_fit.qubit)
        n_qubits_info = len(qubit_info.get('grid_locations', []))
        
        if not (n_qubits_raw == n_qubits_fit == n_qubits_info):
            print(f"⚠️  Warning: Qubit count mismatch - "
                  f"Raw: {n_qubits_raw}, Fit: {n_qubits_fit}, Info: {n_qubits_info}")
        
        # 주파수 범위 검증
        freq_info = experiment_data['metadata'].get('dataset_info', {}).get('frequency_range', {})
        if freq_info:
            # numpy 배열로 변환 후 min/max 계산
            freq_values = ds_raw.full_freq.values
            actual_min = float(np.min(freq_values))
            actual_max = float(np.max(freq_values))
            
            if abs(actual_min - freq_info.get('full_freq_min', actual_min)) > 1e3:  # 1kHz tolerance
                print("⚠️  Warning: Frequency range mismatch in metadata")
        
        # 분석 결과와 피팅 데이터 일관성 확인
        if 'analysis_results' in experiment_data:
            analysis = experiment_data['analysis_results']
            per_qubit = analysis.get('per_qubit', {})
            
            for grid_loc, qubit_analysis in per_qubit.items():
                if 'resonance_frequency_Hz' in qubit_analysis:
                    res_freq = qubit_analysis['resonance_frequency_Hz']
                    if res_freq:
                        freq_values = ds_raw.full_freq.values
                        if freq_values.ndim == 2:
                            # 큐빗별로 다른 주파수인 경우, 해당 큐빗의 주파수 범위 확인
                            qubit_idx = qubit_info['grid_locations'].index(grid_loc)
                            freq_min = float(freq_values[qubit_idx].min())
                            freq_max = float(freq_values[qubit_idx].max())
                        else:
                            # 모든 큐빗이 동일한 주파수인 경우
                            freq_min = float(freq_values.min())
                            freq_max = float(freq_values.max())
                        
                        if not (freq_min <= res_freq <= freq_max):
                            print(f"⚠️  Warning: Resonance frequency for {grid_loc} "
                                  f"is outside measurement range")
    
    def _compute_additional_info(self, experiment_data: Dict) -> Dict:
        """추가 정보 계산"""
        ds_raw = experiment_data['ds_raw']
        ds_fit = experiment_data['ds_fit']
        
        # 주파수 데이터를 numpy 배열로 변환
        freq_values = ds_raw.full_freq.values
        
        # 주파수가 2차원인 경우와 1차원인 경우를 구분하여 처리
        if freq_values.ndim == 2:
            # 각 큐빗마다 다른 주파수 범위를 가진 경우
            # 전체 주파수 범위에서 계산
            center_freq = float(np.mean(freq_values))
            freq_span = float(np.max(freq_values) - np.min(freq_values))
            # 첫 번째 큐빗의 resolution을 대표값으로 사용
            freq_resolution = float(freq_values[0, 1] - freq_values[0, 0])
        else:
            # 모든 큐빗이 동일한 주파수 범위를 가진 경우
            center_freq = float(freq_values.mean())
            freq_span = float(freq_values.max() - freq_values.min())
            freq_resolution = float(freq_values[1] - freq_values[0])
        
        computed_info = {
            'frequency_info': {
                'center_frequency_GHz': center_freq / 1e9,
                'span_MHz': freq_span / 1e6,
                'resolution_kHz': freq_resolution / 1e3
            }
        }
        
        # 신호 품질 지표 계산
        if 'IQ_abs' in ds_raw.data_vars:
            signal_quality = []
            for q_idx in range(len(ds_raw.qubit)):
                iq_data = ds_raw.IQ_abs.isel(qubit=q_idx).values
                
                # SNR 추정 (간단한 방법)
                peak_to_peak = np.max(iq_data) - np.min(iq_data)
                noise_estimate = np.std(iq_data[:10])  # 첫 10개 포인트를 노이즈로 가정
                
                snr = peak_to_peak / noise_estimate if noise_estimate > 0 else 0
                signal_quality.append({
                    'qubit_idx': q_idx,
                    'snr_estimate': float(snr),
                    'peak_amplitude': float(np.max(iq_data)),
                    'base_amplitude': float(np.mean(iq_data[:10]))
                })
            
            computed_info['signal_quality'] = signal_quality
        
        # 큐빗별 주파수 정보 추가 (2D 주파수인 경우)
        if freq_values.ndim == 2:
            per_qubit_freq_info = []
            for q_idx in range(len(ds_raw.qubit)):
                qubit_freqs = freq_values[q_idx]
                per_qubit_freq_info.append({
                    'qubit_idx': q_idx,
                    'freq_min_GHz': float(qubit_freqs.min()) / 1e9,
                    'freq_max_GHz': float(qubit_freqs.max()) / 1e9,
                    'center_freq_GHz': float(qubit_freqs.mean()) / 1e9
                })
            computed_info['per_qubit_frequency_info'] = per_qubit_freq_info
        
        return computed_info
    
    def get_plot_ready_data(self, experiment_data: Dict) -> Dict:
        """
        플로팅을 위한 데이터 준비
        
        Returns
        -------
        Dict
            Plotly Dash에서 바로 사용할 수 있는 형태의 데이터
        """
        ds_raw = experiment_data['ds_raw']
        ds_fit = experiment_data['ds_fit']
        qubit_info = experiment_data['qubit_info']
        
        # full_freq가 2D인지 확인
        freq_values = ds_raw.full_freq.values
        is_2d_freq = freq_values.ndim == 2
        
        plot_data = {
            'qubits': [],
            'is_2d_frequency': is_2d_freq
        }
        
        # 전체 주파수 축 정보 (1D인 경우만)
        if not is_2d_freq:
            plot_data['frequency_axis'] = {
                'full_freq_GHz': (ds_raw.full_freq / 1e9).values.tolist(),
                'detuning_MHz': (ds_raw.detuning / 1e6).values.tolist()
            }
        
        # 각 큐빗별 데이터 준비
        for idx, grid_loc in enumerate(qubit_info['grid_locations']):
            qubit_name = qubit_info['qubit_names'][idx]
            
            # 큐빗별 데이터 선택
            qubit_raw = ds_raw.isel(qubit=idx)
            
            # Raw 데이터
            raw_data = {
                'grid_location': grid_loc,
                'qubit_name': qubit_name,
                'phase': qubit_raw.phase.values.tolist(),
                'IQ_abs': (qubit_raw.IQ_abs / 1e-3).values.tolist(),  # mV 단위
            }
            
            # 2D 주파수인 경우 큐빗별 주파수 축 추가
            if is_2d_freq:
                raw_data['frequency_axis'] = {
                    'full_freq_GHz': (qubit_raw.full_freq / 1e9).values.tolist(),
                    'detuning_MHz': (qubit_raw.detuning / 1e6).values.tolist()
                }
            
            # Fit 데이터
            if all(var in ds_fit.data_vars for var in ['amplitude', 'position', 'width', 'base_line']):
                fit_params = ds_fit.isel(qubit=idx)
                
                # 각 파라미터를 스칼라로 안전하게 변환
                amplitude_val = self._safe_float_conversion(fit_params.amplitude.values)
                position_val = self._safe_float_conversion(fit_params.position.values)
                width_val = self._safe_float_conversion(fit_params.width.values)
                base_line_val = float(fit_params.base_line.mean().values)
                
                # Lorentzian dip 함수로 피팅 곡선 생성
                detuning_values = qubit_raw.detuning.values
                fitted_curve = self._compute_lorentzian_dip(
                    detuning_values,
                    amplitude_val,
                    position_val,
                    width_val / 2,
                    base_line_val
                )
                
                raw_data['fit'] = {
                    'curve': (fitted_curve / 1e-3).tolist(),  # mV 단위
                    'amplitude': amplitude_val,
                    'position': position_val,
                    'width': width_val,
                    'base_line': base_line_val
                }
            
            # 분석 결과 추가
            if 'analysis_results' in experiment_data:
                analysis = experiment_data['analysis_results'].get('per_qubit', {}).get(grid_loc, {})
                raw_data['analysis'] = {
                    'resonance_frequency_GHz': analysis.get('resonance_frequency_Hz', 0) / 1e9,
                    'q_factor': analysis.get('q_factor'),
                    'width_MHz': analysis.get('width_Hz', 0) / 1e6
                }
            
            plot_data['qubits'].append(raw_data)
        
        # 그리드 정보 추가
        plot_data['grid_info'] = qubit_info.get('grid_shape', {})
        
        return plot_data
    
    def _safe_float_conversion(self, value):
        """numpy 값을 안전하게 float로 변환"""
        if hasattr(value, 'ndim'):
            if value.ndim == 0:
                return float(value)
            else:
                return float(value.item())
        return float(value)
    
    def _compute_lorentzian_dip(self, x, amplitude, position, hwhm, base_line):
        """Lorentzian dip 함수 계산"""
        return base_line - amplitude / (1 + ((x - position) / hwhm) ** 2)
    
    def load_multiple_experiments(self, base_dir: Path, 
                                 experiment_type: str = "resonator_spectroscopy",
                                 limit: Optional[int] = None) -> List[Dict]:
        """
        여러 실험 데이터를 한번에 로드
        
        Parameters
        ----------
        base_dir : Path
            실험 데이터들이 있는 기본 디렉토리
        experiment_type : str
            로드할 실험 타입
        limit : int, optional
            로드할 최대 실험 수
            
        Returns
        -------
        List[Dict]
            로드된 실험 데이터 리스트
        """
        experiments = []
        
        # 실험 디렉토리 찾기
        exp_dirs = [d for d in base_dir.iterdir() 
                   if d.is_dir() and d.name.startswith(experiment_type)]
        
        # 시간순 정렬 (최신 순)
        exp_dirs.sort(reverse=True)
        
        # 제한이 있으면 적용
        if limit:
            exp_dirs = exp_dirs[:limit]
        
        print(f"\nFound {len(exp_dirs)} {experiment_type} experiments")
        
        for exp_dir in exp_dirs:
            exp_data = self.load_experiment(exp_dir)
            if exp_data:
                experiments.append(exp_data)
        
        print(f"\nSuccessfully loaded {len(experiments)} experiments")
        
        return experiments
    
    def compare_experiments(self, exp1: Dict, exp2: Dict) -> Dict:
        """
        두 실험 데이터 비교
        
        Returns
        -------
        Dict
            비교 결과
        """
        comparison = {
            'compatible': True,
            'differences': []
        }
        
        # 주파수 범위 비교 - numpy 배열로 변환
        freq1_values = exp1['ds_raw'].full_freq.values
        freq2_values = exp2['ds_raw'].full_freq.values
        
        # 전체 범위로 비교 (2D든 1D든)
        freq1_min = float(np.min(freq1_values))
        freq1_max = float(np.max(freq1_values))
        freq2_min = float(np.min(freq2_values))
        freq2_max = float(np.max(freq2_values))
        
        if abs(freq1_min - freq2_min) > 1e6 or abs(freq1_max - freq2_max) > 1e6:  # 1MHz tolerance
            comparison['differences'].append('frequency_range')
            comparison['compatible'] = False
        
        # 큐빗 수 비교
        if len(exp1['ds_raw'].qubit) != len(exp2['ds_raw'].qubit):
            comparison['differences'].append('qubit_count')
            comparison['compatible'] = False
        
        # 그리드 위치 비교
        grid1 = set(exp1['qubit_info']['grid_locations'])
        grid2 = set(exp2['qubit_info']['grid_locations'])
        
        if grid1 != grid2:
            comparison['differences'].append('grid_locations')
            comparison['grid_diff'] = {
                'only_in_exp1': list(grid1 - grid2),
                'only_in_exp2': list(grid2 - grid1)
            }
        
        return comparison


# 사용 예시 함수들
def load_latest_resonator_spec(base_dir: str = "D:/Codes/Career/Kyunghoon/Playground/HI_16Jun2025/calibration_dashboard/dashboard_data") -> Optional[Dict]:
    """최신 Resonator Spectroscopy 실험 데이터 로드"""
    loader = ResonatorSpecDataLoader()
    base_path = Path(base_dir)
    
    # 최신 실험 찾기
    exp_dirs = [d for d in base_path.iterdir() 
               if d.is_dir() and d.name.startswith("resonator_spectroscopy")]
    
    if not exp_dirs:
        print("No resonator spectroscopy experiments found")
        return None
    
    # 최신 실험 로드
    latest_dir = max(exp_dirs, key=lambda d: d.stat().st_mtime)
    return loader.load_experiment(latest_dir)


def load_and_prepare_for_plotting(experiment_dir: Path) -> Optional[Dict]:
    """플로팅을 위한 데이터 로드 및 준비"""
    loader = ResonatorSpecDataLoader()
    
    # 실험 로드
    exp_data = loader.load_experiment(experiment_dir)
    if not exp_data:
        return None
    
    # 플롯 데이터 준비
    plot_data = loader.get_plot_ready_data(exp_data)
    
    return {
        'experiment': exp_data,
        'plot_data': plot_data
    }