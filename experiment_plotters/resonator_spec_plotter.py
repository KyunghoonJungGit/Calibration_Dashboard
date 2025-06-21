"""
Resonator Spectroscopy Plotter Module
Resonator Spectroscopy 실험 데이터 시각화를 위한 플로터
"""
from typing import Dict, List, Any, Tuple
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc
from .base_plotter import ExperimentPlotter


class ResonatorSpecPlotter(ExperimentPlotter):
    """Resonator Spectroscopy 실험 플로터"""
    
    def __init__(self):
        """Resonator Spec 플로터 초기화"""
        super().__init__()
        
        # Resonator Spec 특화 설정
        self.trace_colors = {
            'phase': '#1f77b4',      # 파란색
            'amplitude': '#2ca02c',   # 초록색
            'fit': '#d62728'         # 빨간색
        }
        
        # 플롯 기본 설정
        self.phase_range = (-0.2, 0.2)  # Phase 기본 범위 (rad)
        self.amplitude_unit = 1e-3      # mV 단위 변환
    
    @property
    def experiment_type(self) -> str:
        """실험 타입"""
        return "resonator_spectroscopy"
    
    def get_display_options(self) -> List[Any]:
        """Resonator Spec 전용 디스플레이 옵션"""
        return [
            # 플롯 타입 선택
            html.Div([
                html.Label("Plot Type:", className="font-weight-bold mb-2"),
                dcc.RadioItems(
                    id={'type': 'res-spec-option', 'index': 0},
                    options=[
                        {"label": "Amplitude", "value": "amplitude"},
                        {"label": "Phase", "value": "phase"},
                        {"label": "Both", "value": "both"}
                    ],
                    value="amplitude",
                    className="mb-3",
                    labelStyle={'display': 'block', 'margin-bottom': '5px'}
                )
            ]),
            
            html.Hr(),
            
            # 추가 옵션들
            html.Div([
                html.Label("Display Options:", className="font-weight-bold mb-2"),
                dcc.Checklist(
                    id={'type': 'res-spec-option', 'index': 1},
                    options=[
                        {"label": "Show Fit", "value": "show_fit"},
                        {"label": "Show Resonance Line", "value": "show_resonance"},
                        {"label": "Show Dual X-axis", "value": "dual_axis"},
                        {"label": "Auto-scale Y-axis", "value": "auto_scale"}
                    ],
                    value=["show_fit", "show_resonance", "dual_axis", "auto_scale"],
                    className="mb-3",
                    labelStyle={'display': 'block', 'margin-bottom': '5px'}
                )
            ]),
            
            html.Hr(),
            
            # 레이아웃 옵션
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Label("Max Columns:", className="mb-2"),
                        dcc.Dropdown(
                            id={'type': 'res-spec-option', 'index': 2},
                            options=[
                                {"label": "1", "value": "1"},
                                {"label": "2", "value": "2"},
                                {"label": "3", "value": "3"},
                                {"label": "4", "value": "4"}
                            ],
                            value="2",
                            style={"width": "100%"}
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Height per subplot:", className="mb-2"),
                        dcc.Input(
                            id={'type': 'res-spec-option', 'index': 3},
                            type="number",
                            value=350,
                            min=200,
                            max=600,
                            step=50,
                            style={"width": "100%"}
                        )
                    ], width=6)
                ])
            ]),
            
            html.Hr(),
            
            # 분석 정보 표시
            html.Div([
                html.Label("Analysis Info:", className="font-weight-bold mb-2"),
                html.Div(id='res-spec-analysis-info', className="small text-muted")
            ])
        ]
    
    def get_default_options(self) -> Dict:
        """기본 옵션"""
        return {
            'plot_type': 'amplitude',
            'show_options': ['show_fit', 'show_resonance', 'dual_axis', 'auto_scale'],
            'max_cols': '2',
            'subplot_height': 350
        }
    
    def create_plot(self, exp_data: Dict, selected_qubits: List[str], 
                   plot_options: Dict) -> go.Figure:
        """Resonator Spec 플롯 생성"""
        # 옵션 추출
        plot_type = plot_options.get('plot_type', 'amplitude')
        show_options = plot_options.get('show_options', ['show_fit', 'show_resonance', 'dual_axis', 'auto_scale'])
        show_fit = 'show_fit' in show_options
        show_resonance = 'show_resonance' in show_options
        dual_axis = 'dual_axis' in show_options
        auto_scale = 'auto_scale' in show_options
        
        # 동적 레이아웃 설정
        max_cols = int(plot_options.get('max_cols', '2'))
        subplot_height = int(plot_options.get('subplot_height', 350))
        
        # 원래 설정 임시 변경
        original_max_cols = self.MAX_COLS
        original_subplot_height = self.SUBPLOT_HEIGHT
        self.MAX_COLS = max_cols
        self.SUBPLOT_HEIGHT = subplot_height
        
        try:
            # 데이터 추출
            ds_raw = exp_data['ds_raw']
            ds_fit = exp_data['ds_fit']
            qubit_info = exp_data['qubit_info']
            
            n_qubits = len(selected_qubits)
            
            # 레이아웃 계산
            rows_multiplier = 2 if plot_type == 'both' else 1
            rows, cols, total_height = self.get_subplot_layout(n_qubits, rows_multiplier)
            
            # 서브플롯 타이틀 생성
            subplot_titles = self._generate_subplot_titles(selected_qubits, plot_type)
            
            # Figure 생성
            fig = self.create_subplots_figure(rows, cols, subplot_titles, total_height)
            
            # 플롯 데이터 추가
            self._add_resonator_spec_data(
                fig, ds_raw, ds_fit, qubit_info,
                selected_qubits, plot_type, rows, cols,
                show_fit, show_resonance, dual_axis, auto_scale,
                exp_data
            )
            
            # 축 레이블 업데이트
            if plot_type in ['amplitude', 'both']:
                fig.update_xaxes(title_text="RF frequency [GHz]")
                fig.update_yaxes(title_text="R = √(I² + Q²) [mV]")
            
            if plot_type == 'phase':
                fig.update_xaxes(title_text="RF frequency [GHz]")
                fig.update_yaxes(title_text="Phase [rad]")
            
            # 서브플롯 타이틀 위치 조정
            self.update_subplot_titles_position(fig, subplot_titles)
            
            # 공통 레이아웃 적용
            title = f"Resonator Spectroscopy - {plot_type.capitalize()}"
            self.apply_common_layout(fig, title)
            
            return fig
            
        finally:
            # 원래 설정 복원
            self.MAX_COLS = original_max_cols
            self.SUBPLOT_HEIGHT = original_subplot_height
    
    def _generate_subplot_titles(self, selected_qubits: List[str], 
                               plot_type: str) -> List[str]:
        """서브플롯 타이틀 생성"""
        if plot_type == 'both':
            titles = []
            for loc in selected_qubits:
                titles.append(f"{loc} - Amplitude")
            for loc in selected_qubits:
                titles.append(f"{loc} - Phase")
        else:
            titles = [f"qubit = {loc}" for loc in selected_qubits]
        
        return titles
    
    def _add_resonator_spec_data(self, fig: go.Figure, ds_raw: Any, ds_fit: Any,
                                qubit_info: Dict, selected_qubits: List[str],
                                plot_type: str, rows: int, cols: int,
                                show_fit: bool, show_resonance: bool, 
                                dual_axis: bool, auto_scale: bool,
                                exp_data: Dict):
        """Resonator Spec 데이터를 figure에 추가"""
        # 범례 표시 여부 추적
        show_legends = {
            'amplitude': True,
            'phase': True,
            'fit': True
        }
        
        # 주파수가 2D인지 확인
        freq_values = ds_raw.full_freq.values
        is_2d_freq = freq_values.ndim == 2
        
        for idx, grid_location in enumerate(selected_qubits):
            col = (idx % cols) + 1
            
            try:
                # 큐빗 데이터 가져오기
                qubit_data, qubit_name = self.get_qubit_data(
                    {'ds_raw': ds_raw}, grid_location, qubit_info
                )
                
                # 주파수 축 데이터 가져오기
                if is_2d_freq:
                    # 큐빗별로 다른 주파수를 가진 경우
                    qubit_idx = qubit_info['qubit_mapping'][grid_location]['dataset_index']
                    full_freq_GHz = freq_values[qubit_idx] / 1e9  # GHz 단위
                    detuning_MHz = ds_raw.detuning.isel(qubit=qubit_idx).values / 1e6  # MHz 단위
                else:
                    # 모든 큐빗이 동일한 주파수를 가진 경우
                    full_freq_GHz = freq_values / 1e9  # GHz 단위
                    detuning_MHz = ds_raw.detuning.values / 1e6    # MHz 단위
                
                # Amplitude plot
                if plot_type in ['amplitude', 'both']:
                    row = (idx // cols) + 1
                    self._add_amplitude_plot(
                        fig, qubit_data, ds_fit, qubit_name,
                        full_freq_GHz, detuning_MHz,
                        row, col, show_legends, 
                        show_fit, show_resonance, dual_axis, auto_scale
                    )
                
                # Phase plot
                if plot_type in ['phase', 'both']:
                    if plot_type == 'both':
                        row = rows // 2 + (idx // cols) + 1
                    else:
                        row = (idx // cols) + 1
                    
                    self._add_phase_plot(
                        fig, qubit_data,
                        full_freq_GHz, detuning_MHz,
                        row, col, show_legends,
                        dual_axis, auto_scale
                    )
                    
            except Exception as e:
                print(f"Error plotting {grid_location}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _add_amplitude_plot(self, fig: go.Figure, qubit_data: Any, ds_fit: Any,
                           qubit_name: str, full_freq_GHz: Any, detuning_MHz: Any,
                           row: int, col: int, show_legends: Dict,
                           show_fit: bool, show_resonance: bool, 
                           dual_axis: bool, auto_scale: bool):
        """Amplitude 플롯 추가"""
        # IQ_abs 데이터 (mV 단위로 변환)
        iq_abs_mV = qubit_data.IQ_abs.values / self.amplitude_unit
        
        # 메인 amplitude trace
        fig.add_trace(
            go.Scatter(
                x=full_freq_GHz,
                y=iq_abs_mV,
                name='Amplitude' if show_legends['amplitude'] else None,
                line=dict(color=self.trace_colors['amplitude'], width=2),
                showlegend=show_legends['amplitude'],
                legendgroup='amplitude',
                hovertemplate='%{x:.6f} GHz<br>%{y:.3f} mV<extra></extra>'
            ),
            row=row, col=col
        )
        show_legends['amplitude'] = False
        
        # Fit 추가
        if show_fit:
            try:
                # 피팅 파라미터 가져오기
                qubit_fit = ds_fit.sel(qubit=qubit_name)
                
                # 각 파라미터를 스칼라로 안전하게 변환
                amplitude = self._safe_float_conversion(qubit_fit.amplitude.values)
                position = self._safe_float_conversion(qubit_fit.position.values)
                width = self._safe_float_conversion(qubit_fit.width.values)
                base_line = float(qubit_fit.base_line.mean().values)
                
                # Lorentzian dip 계산
                detuning = qubit_data.detuning.values
                fitted_curve = self._compute_lorentzian_dip(
                    detuning, amplitude, position, width/2, base_line
                )
                fitted_curve_mV = fitted_curve / self.amplitude_unit
                
                # Fit trace 추가
                fig.add_trace(
                    go.Scatter(
                        x=full_freq_GHz,
                        y=fitted_curve_mV,
                        name='Fit' if show_legends['fit'] else None,
                        line=dict(color=self.trace_colors['fit'], width=2, dash='dash'),
                        showlegend=show_legends['fit'],
                        legendgroup='fit',
                        hovertemplate='Fit<br>%{x:.6f} GHz<br>%{y:.3f} mV<extra></extra>'
                    ),
                    row=row, col=col
                )
                show_legends['fit'] = False
                
                # 공진 주파수 표시 (피팅 결과에서 직접 계산)
                if show_resonance:
                    # 중심 주파수 계산
                    center_freq = np.mean(qubit_data.full_freq.values)
                    resonance_freq = center_freq + position
                    res_freq_GHz = resonance_freq / 1e9
                    
                    fig.add_vline(
                        x=res_freq_GHz,
                        line_dash="dot",
                        line_color="black",
                        line_width=1,
                        annotation_text=f"{res_freq_GHz:.6f} GHz" if row == 1 and col == 1 else None,
                        row=row, col=col
                    )
                
            except Exception as e:
                print(f"Warning: Could not add fit for {qubit_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Y축 범위 설정
        if auto_scale:
            y_min, y_max = self.calculate_y_range(iq_abs_mV, margin_ratio=0.1)
        else:
            # 데이터 범위에 맞춰 설정
            y_min = np.min(iq_abs_mV) * 0.9
            y_max = np.max(iq_abs_mV) * 1.1
        
        fig.update_yaxes(range=[y_min, y_max], row=row, col=col)
        
        # Dual X-axis 설정
        if dual_axis and row == 1:
            # Detuning 축 추가 (상단)
            fig.update_xaxes(
                title_text="Detuning [MHz]",
                secondary_y=False,
                overlaying="x",
                side="top",
                row=row, col=col
            )
    
    def _add_phase_plot(self, fig: go.Figure, qubit_data: Any,
                       full_freq_GHz: Any, detuning_MHz: Any,
                       row: int, col: int, show_legends: Dict,
                       dual_axis: bool, auto_scale: bool):
        """Phase 플롯 추가"""
        # Phase 데이터
        phase_rad = qubit_data.phase.values
        
        # Phase trace
        fig.add_trace(
            go.Scatter(
                x=full_freq_GHz,
                y=phase_rad,
                name='Phase' if show_legends['phase'] else None,
                line=dict(color=self.trace_colors['phase'], width=2),
                showlegend=show_legends['phase'],
                legendgroup='phase',
                hovertemplate='%{x:.6f} GHz<br>%{y:.3f} rad<extra></extra>'
            ),
            row=row, col=col
        )
        show_legends['phase'] = False
        
        # Y축 범위 설정
        if auto_scale:
            y_min, y_max = self.calculate_y_range(phase_rad, margin_ratio=0.1)
        else:
            y_min, y_max = self.phase_range
        
        fig.update_yaxes(range=[y_min, y_max], row=row, col=col)
        
        # Dual X-axis 설정
        if dual_axis and row == 1:
            # Detuning 축 추가 (상단)
            fig.update_xaxes(
                title_text="Detuning [MHz]",
                secondary_y=False,
                overlaying="x",
                side="top",
                row=row, col=col
            )
    
    def _safe_float_conversion(self, value):
        """numpy 값을 안전하게 float로 변환"""
        if hasattr(value, 'ndim'):
            if value.ndim == 0:
                return float(value)
            else:
                return float(value.item())
        return float(value)
    
    def _compute_lorentzian_dip(self, x: np.ndarray, amplitude: float, 
                               position: float, hwhm: float, base_line: float) -> np.ndarray:
        """Lorentzian dip 함수 계산
        
        Parameters
        ----------
        x : np.ndarray
            Detuning values
        amplitude : float
            Dip amplitude
        position : float
            Center position
        hwhm : float
            Half width at half maximum
        base_line : float
            Baseline level
            
        Returns
        -------
        np.ndarray
            Fitted curve values
        """
        return base_line - amplitude / (1 + ((x - position) / hwhm) ** 2)