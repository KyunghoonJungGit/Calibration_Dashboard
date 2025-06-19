"""
Resonator Spectroscopy Data Saver Module
Save Resonator Spectroscopy experiment data for Plotly Dash dashboard
공진기 분광 실험 데이터를 Dash 대시보드용으로 저장하는 모듈
"""

import json
import xarray as xr
from pathlib import Path
from datetime import datetime
from typing import List, Any, Dict, Optional
import numpy as np


def save_resonator_spec_experiment_for_dashboard(
    ds_raw: xr.Dataset, 
    qubits: List[Any], 
    ds_fit: xr.Dataset,
    experiment_type: str = "resonator_spectroscopy",
    base_dir: str = "D:/Codes/Career/Kyunghoon/Playground/HI_16Jun2025/calibration_dashboard/dashboard_data",
    additional_info: Optional[Dict] = None
) -> Path:
    """
    Resonator Spectroscopy 실험 데이터를 Dash 대시보드용으로 저장
    
    Parameters
    ----------
    ds_raw : xr.Dataset
        Raw experimental data containing phase and IQ_abs data
        Expected coordinates: full_freq, detuning, qubit
        Expected data variables: phase, IQ_abs
    qubits : List[AnyTransmon]
        List of qubit objects with grid_location attributes
    ds_fit : xr.Dataset
        Fitted parameters from Lorentzian dip fitting
        Expected data variables: amplitude, position, width, base_line
    experiment_type : str
        Type of experiment (default: "resonator_spectroscopy")
    base_dir : str
        Base directory for saving data
    additional_info : Dict, optional
        Additional information to save (e.g., experiment parameters)
    
    Returns
    -------
    Path
        Directory where data was saved
    """
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{experiment_type}_{timestamp}"
    
    # 저장 디렉토리 생성
    save_dir = Path(base_dir) / experiment_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== Saving Resonator Spectroscopy experiment data for dashboard ===")
    print(f"Experiment ID: {experiment_id}")
    print(f"Save directory: {save_dir}")
    
    # 1. xarray 데이터셋 저장
    ds_raw_path = save_dir / "ds_raw.nc"
    ds_fit_path = save_dir / "ds_fit.nc"
    
    # 데이터 유효성 검사
    _validate_resonator_spec_data(ds_raw, ds_fit)
    
    ds_raw.to_netcdf(ds_raw_path)
    ds_fit.to_netcdf(ds_fit_path)
    print(f"✓ Saved raw data: {ds_raw_path}")
    print(f"✓ Saved fit data: {ds_fit_path}")
    
    # 2. Qubit 정보 추출 및 저장
    qubit_info = _extract_qubit_info_for_resonator_spec(qubits, ds_raw)
    qubit_info_path = save_dir / "qubit_info.json"
    
    with open(qubit_info_path, 'w') as f:
        json.dump(qubit_info, f, indent=2)
    print(f"✓ Saved qubit info: {qubit_info_path}")
    
    # 3. 실험 특화 분석 정보 저장
    analysis_results = _extract_resonator_spec_analysis(ds_raw, ds_fit, qubits)
    analysis_path = save_dir / "analysis_results.json"
    
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"✓ Saved analysis results: {analysis_path}")
    
    # 4. 메타데이터 저장
    metadata = {
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "timestamp": timestamp,
        "timestamp_full": datetime.now().isoformat(),
        "data_files": {
            "ds_raw": "ds_raw.nc",
            "ds_fit": "ds_fit.nc",
            "qubit_info": "qubit_info.json",
            "analysis_results": "analysis_results.json"
        },
        "dataset_info": {
            "dimensions": dict(ds_raw.dims),
            "coordinates": list(ds_raw.coords),
            "data_vars": list(ds_raw.data_vars),
            "qubit_count": len(qubits),
            "frequency_range": {
                "full_freq_min": float(ds_raw.full_freq.min().values),
                "full_freq_max": float(ds_raw.full_freq.max().values),
                "detuning_min": float(ds_raw.detuning.min().values),
                "detuning_max": float(ds_raw.detuning.max().values),
                "n_frequency_points": len(ds_raw.full_freq)
            }
        }
    }
    
    # 추가 정보가 있으면 메타데이터에 포함
    if additional_info:
        metadata["experiment_parameters"] = additional_info
    
    metadata_path = save_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")
    
    # 5. 실험 설정 정보 저장 (선택적)
    if additional_info:
        config_path = save_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(additional_info, f, indent=2)
        print(f"✓ Saved experiment config: {config_path}")
    
    # 6. 완료 플래그 생성
    complete_flag = save_dir / ".complete"
    complete_flag.touch()
    
    print(f"\n✓ All Resonator Spectroscopy data saved successfully to: {save_dir}")
    print("=" * 50)
    
    return save_dir


def _validate_resonator_spec_data(ds_raw: xr.Dataset, ds_fit: xr.Dataset):
    """Resonator Spectroscopy 데이터 유효성 검사"""
    
    # Raw 데이터 검사
    required_coords = ['full_freq', 'detuning', 'qubit']
    required_vars = ['phase', 'IQ_abs']
    
    missing_coords = [c for c in required_coords if c not in ds_raw.coords]
    if missing_coords:
        raise ValueError(f"Missing required coordinates in ds_raw: {missing_coords}")
    
    missing_vars = [v for v in required_vars if v not in ds_raw.data_vars]
    if missing_vars:
        raise ValueError(f"Missing required data variables in ds_raw: {missing_vars}")
    
    # Fit 데이터 검사
    fit_vars = ['amplitude', 'position', 'width', 'base_line']
    missing_fit_vars = [v for v in fit_vars if v not in ds_fit.data_vars]
    if missing_fit_vars:
        print(f"⚠️  Warning: Missing fit parameters: {missing_fit_vars}")


def _extract_qubit_info_for_resonator_spec(qubits: List[Any], ds_raw: xr.Dataset) -> Dict:
    """
    Resonator Spectroscopy용 Qubit 정보 추출
    
    Returns
    -------
    Dict
        {
            "grid_locations": ["0,7", "0,6", ...],
            "qubit_names": ["q0-7", "q0-6", ...],
            "qubit_mapping": {
                "0,7": {
                    "dataset_index": 0, 
                    "qubit_name": "q0-7",
                    "grid_row": 7,
                    "grid_col": 0
                },
                ...
            },
            "dataset_qubit_dim": "qubit",
            "grid_shape": {"rows": 8, "cols": 1}
        }
    """
    
    grid_locations = []
    qubit_names = []
    qubit_mapping = {}
    grid_rows = []
    grid_cols = []
    
    # 데이터셋의 qubit 차원 정보
    dataset_qubit_dim = ds_raw.qubit.name if hasattr(ds_raw.qubit, 'name') else 'qubit'
    dataset_qubit_values = list(ds_raw.qubit.values)
    
    for idx, q in enumerate(qubits):
        # grid_location 추출
        if hasattr(q, 'grid_location'):
            grid_loc = q.grid_location
        elif hasattr(q, 'name') and ',' in str(q.name):
            grid_loc = q.name
        else:
            # fallback: 인덱스 기반 생성
            grid_loc = f"0,{idx}"
        
        grid_locations.append(grid_loc)
        
        # 그리드 좌표 파싱
        try:
            col, row = map(int, grid_loc.split(','))
            grid_rows.append(row)
            grid_cols.append(col)
        except:
            grid_rows.append(idx)
            grid_cols.append(0)
        
        # 데이터셋에서의 qubit 이름 확인
        if idx < len(dataset_qubit_values):
            qubit_name = str(dataset_qubit_values[idx])
        else:
            qubit_name = f"q{grid_loc}"
        
        qubit_names.append(qubit_name)
        
        # 매핑 정보 저장
        qubit_mapping[grid_loc] = {
            "dataset_index": idx,
            "qubit_name": qubit_name,
            "grid_location": grid_loc,
            "grid_row": grid_rows[-1],
            "grid_col": grid_cols[-1]
        }
    
    # 그리드 형태 계산
    grid_shape = {
        "rows": max(grid_rows) - min(grid_rows) + 1 if grid_rows else 1,
        "cols": max(grid_cols) - min(grid_cols) + 1 if grid_cols else 1,
        "min_row": min(grid_rows) if grid_rows else 0,
        "min_col": min(grid_cols) if grid_cols else 0
    }
    
    return {
        "grid_locations": grid_locations,
        "qubit_names": qubit_names,
        "qubit_mapping": qubit_mapping,
        "dataset_qubit_dim": dataset_qubit_dim,
        "grid_shape": grid_shape
    }


def _extract_resonator_spec_analysis(ds_raw: xr.Dataset, ds_fit: xr.Dataset, qubits: List[Any]) -> Dict:
    """
    Resonator Spectroscopy 분석 결과 추출
    각 큐빗별 공진 주파수, Q factor 등의 핵심 정보를 저장
    """
    
    analysis_results = {
        "summary": {},
        "per_qubit": {}
    }
    
    # 전체 요약 정보
    analysis_results["summary"] = {
        "total_qubits": len(qubits),
        "frequency_span_Hz": float(ds_raw.full_freq.max().values - ds_raw.full_freq.min().values),
        "frequency_points": len(ds_raw.full_freq),
        "measurement_type": "resonator_spectroscopy"
    }
    
    # 각 큐빗별 분석 결과
    for idx, q in enumerate(qubits):
        grid_loc = q.grid_location if hasattr(q, 'grid_location') else f"0,{idx}"
        qubit_name = str(ds_raw.qubit.values[idx]) if idx < len(ds_raw.qubit) else f"q{idx}"
        
        # 피팅 결과 추출
        qubit_fit = ds_fit.isel(qubit=idx)
        
        # 공진 주파수 계산 (position은 detuning 기준)
        if 'position' in qubit_fit.data_vars:
            detuning_at_resonance = float(qubit_fit.position.values)
            # 중심 주파수 찾기
            center_freq_idx = len(ds_raw.full_freq) // 2
            center_freq = float(ds_raw.full_freq.isel(frequency=center_freq_idx).values)
            resonance_freq = center_freq + detuning_at_resonance
        else:
            resonance_freq = None
            detuning_at_resonance = None
        
        # Q factor 계산 (간단한 추정)
        if 'width' in qubit_fit.data_vars and resonance_freq:
            width_hz = float(qubit_fit.width.values)
            q_factor = resonance_freq / width_hz if width_hz > 0 else None
        else:
            q_factor = None
            width_hz = None
        
        # 진폭 정보
        if 'amplitude' in qubit_fit.data_vars:
            amplitude = float(qubit_fit.amplitude.values)
        else:
            amplitude = None
        
        # 베이스라인
        if 'base_line' in qubit_fit.data_vars:
            base_line = float(qubit_fit.base_line.mean().values)
        else:
            base_line = None
        
        analysis_results["per_qubit"][grid_loc] = {
            "qubit_name": qubit_name,
            "grid_location": grid_loc,
            "resonance_frequency_Hz": resonance_freq,
            "detuning_at_resonance_Hz": detuning_at_resonance,
            "width_Hz": width_hz,
            "q_factor": q_factor,
            "amplitude": amplitude,
            "base_line": base_line,
            "fit_quality": {
                "has_valid_fit": all(v is not None for v in [resonance_freq, width_hz, amplitude]),
                "resonance_in_range": (
                    float(ds_raw.full_freq.min().values) < resonance_freq < float(ds_raw.full_freq.max().values)
                ) if resonance_freq else False
            }
        }
    
    # 통계 정보 추가
    valid_q_factors = [
        res["q_factor"] for res in analysis_results["per_qubit"].values() 
        if res["q_factor"] is not None
    ]
    
    if valid_q_factors:
        analysis_results["summary"]["q_factor_stats"] = {
            "mean": float(np.mean(valid_q_factors)),
            "std": float(np.std(valid_q_factors)),
            "min": float(np.min(valid_q_factors)),
            "max": float(np.max(valid_q_factors))
        }
    
    return analysis_results


# 실험 스크립트에서 사용할 수 있는 wrapper 함수
def save_resonator_spec_to_dashboard(node, base_dir: Optional[str] = None):
    """
    Qualibration node에서 직접 호출할 수 있는 wrapper 함수
    
    Usage in experiment script:
    ```python
    @node.run_action(skip_if=node.parameters.simulate)
    def plot_data(node: QualibrationNode[Parameters, Quam]):
        # ... existing plotting code ...
        
        # Save for dashboard
        if node.parameters.get('save_for_dashboard', True):
            from resonator_spec_data_saver import save_resonator_spec_to_dashboard
            save_resonator_spec_to_dashboard(node)
    ```
    """
    
    # 실험 파라미터 추출
    additional_info = {}
    if hasattr(node, 'parameters'):
        param_dict = {}
        for attr in dir(node.parameters):
            if not attr.startswith('_'):
                value = getattr(node.parameters, attr)
                # JSON 직렬화 가능한 타입만 저장
                if isinstance(value, (str, int, float, bool, list, dict)):
                    param_dict[attr] = value
        additional_info['parameters'] = param_dict
    
    # 실험 정보 추가
    if hasattr(node, 'namespace'):
        additional_info['experiment_info'] = {
            'num_averages': node.namespace.get('num_averages', None),
            'readout_amplitude': node.namespace.get('readout_amplitude', None),
            'readout_duration': node.namespace.get('readout_duration', None)
        }
    
    # 데이터 저장
    return save_resonator_spec_experiment_for_dashboard(
        ds_raw=node.results["ds_raw"],
        qubits=node.namespace["qubits"],
        ds_fit=node.results["ds_fit"],
        experiment_type="resonator_spectroscopy",
        base_dir=base_dir,
        additional_info=additional_info
    )