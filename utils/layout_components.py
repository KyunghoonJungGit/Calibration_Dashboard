"""
Layout Components Module
대시보드의 공통 레이아웃 컴포넌트들을 제공하는 모듈
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from typing import List, Dict, Any


class LayoutComponents:
    """대시보드 레이아웃 컴포넌트 팩토리"""
    
    def __init__(self):
        """레이아웃 컴포넌트 초기화"""
        self.theme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def create_header(self) -> dbc.Row:
        """대시보드 헤더 생성"""
        return dbc.Row([
            dbc.Col([
                html.H1(
                    "Calibration Experiments Dashboard",
                    className="text-center mb-4",
                    style={'color': self.theme['primary']}
                ),
                html.P(
                    "Real-time monitoring and visualization of quantum calibration experiments",
                    className="text-center text-muted",
                    style={'fontSize': '1.1em'}
                )
            ], width=12)
        ], className="mb-4")
    
    def create_status_alert(self) -> dbc.Row:
        """상태 알림 영역 생성"""
        return dbc.Row([
            dbc.Col([
                dbc.Alert(
                    id='status-alert',
                    color="info",
                    dismissable=True,
                    is_open=False,
                    duration=5000  # 5초 후 자동 닫힘
                )
            ], width=12)
        ], className="mb-3")
    
    def create_experiment_selector_card(self) -> dbc.Card:
        """실험 선택 카드 생성"""
        return dbc.Card([
            dbc.CardBody([
                html.H5(
                    "📋 Experiment Selection", 
                    className="card-title mb-3"
                ),
                
                # 실험 선택 드롭다운
                dcc.Dropdown(
                    id='experiment-selector',
                    options=[],
                    value=None,
                    placeholder="Select an experiment",
                    className="mb-3",
                    style={'fontSize': '0.95em'}
                ),
                
                # 실험 정보 표시
                html.Div(
                    id='experiment-info',
                    className="text-muted mb-3",
                    style={'fontSize': '0.9em'}
                ),
                
                html.Hr(),
                
                # 통계 정보
                dbc.Row([
                    dbc.Col([
                        html.Small("Total experiments: ", className="text-muted"),
                        html.Small(
                            id='experiment-count',
                            className="text-muted font-weight-bold"
                        )
                    ], width=12)
                ], className="mb-2"),
                
                # 컨트롤 버튼
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button(
                            "🔄 Refresh",
                            id="refresh-button",
                            color="primary",
                            size="sm",
                            outline=True
                        ),
                        dbc.Button(
                            "📊 Stats",
                            id="stats-button",
                            color="info",
                            size="sm",
                            outline=True,
                            disabled=True  # 추후 구현
                        )
                    ]),
                    html.Small(
                        id="last-update-time",
                        className="text-muted ms-2 d-block mt-2"
                    )
                ])
            ])
        ], className="h-100")
    
    def create_plot_area(self) -> dbc.Row:
        """메인 플롯 영역 생성"""
        return dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-main",
                    type="default",
                    color=self.theme['primary'],
                    children=[
                        dcc.Graph(
                            id='main-plot',
                            style={'height': '80vh'},
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'calibration_plot',
                                    'height': 1080,
                                    'width': 1920,
                                    'scale': 2
                                }
                            }
                        )
                    ]
                )
            ], width=12)
        ], className="mb-4")
    
    def create_data_stores(self) -> List[dcc.Store]:
        """데이터 저장소 컴포넌트들 생성"""
        return [
            # 현재 실험 데이터
            dcc.Store(
                id='current-experiment-data',
                storage_type='memory',
                data=None
            ),
            
            # 전체 실험 목록
            dcc.Store(
                id='experiments-store',
                storage_type='memory',
                data={}
            ),
            
            # 새 실험 플래그
            dcc.Store(
                id='new-experiments-flag',
                storage_type='memory',
                data={'has_new': False, 'count': 0}
            )
        ]
    
    def create_experiment_card(self, exp_id: str, exp_data: Dict) -> dbc.Card:
        """개별 실험 정보 카드 생성 (추후 확장용)"""
        exp_type = exp_data.get('type', 'Unknown')
        timestamp = exp_data.get('timestamp', 'N/A')
        
        # 실험 타입별 색상
        type_colors = {
            'time_of_flight': 'primary',
            'resonator_spectroscopy': 'info',
            'qubit_spectroscopy': 'success',
            'ramsey': 'warning',
            'rabi_amplitude': 'danger',
            'rabi_power': 'secondary'
        }
        
        color = type_colors.get(exp_type, 'secondary')
        
        return dbc.Card([
            dbc.CardHeader([
                dbc.Badge(
                    exp_type.replace('_', ' ').title(),
                    color=color,
                    className="me-2"
                ),
                html.Small(timestamp, className="text-muted")
            ]),
            dbc.CardBody([
                html.P(f"ID: {exp_id}", className="mb-1"),
                html.Small(
                    f"Qubits: {len(exp_data.get('grid_locations', []))}",
                    className="text-muted"
                )
            ])
        ], className="mb-2")
    
    def create_loading_spinner(self, text: str = "Loading...") -> html.Div:
        """로딩 스피너 생성"""
        return html.Div([
            dbc.Spinner(
                color="primary",
                size="lg",
                children=[
                    html.Div(text, className="mt-3 text-muted")
                ]
            )
        ], className="text-center p-5")
    
    def create_error_message(self, error_text: str) -> dbc.Alert:
        """에러 메시지 생성"""
        return dbc.Alert([
            html.H4("⚠️ Error", className="alert-heading"),
            html.P(error_text),
            html.Hr(),
            html.P(
                "Please check the console for more details.",
                className="mb-0"
            )
        ], color="danger", dismissable=True)
    
    def create_info_tooltip(self, tooltip_id: str, tooltip_text: str, 
                          icon: str = "ℹ️") -> html.Span:
        """정보 툴팁 생성"""
        return html.Span([
            html.Span(
                icon,
                id=tooltip_id,
                style={
                    'cursor': 'help',
                    'textDecoration': 'underline',
                    'textDecorationStyle': 'dotted'
                }
            ),
            dbc.Tooltip(
                tooltip_text,
                target=tooltip_id,
                placement="top"
            )
        ])
    
    def create_collapsible_section(self, section_id: str, title: str, 
                                  content: Any, is_open: bool = True) -> html.Div:
        """접을 수 있는 섹션 생성"""
        return html.Div([
            dbc.Button(
                [
                    html.Span("▼" if is_open else "▶", className="me-2"),
                    title
                ],
                id=f"{section_id}-toggle",
                color="link",
                className="text-start w-100 mb-2",
                style={'textDecoration': 'none'}
            ),
            dbc.Collapse(
                content,
                id=f"{section_id}-collapse",
                is_open=is_open
            )
        ])
    
    def create_metric_card(self, title: str, value: Any, 
                          subtitle: str = None, color: str = "primary") -> dbc.Card:
        """메트릭 표시 카드 생성"""
        return dbc.Card([
            dbc.CardBody([
                html.H6(title, className="text-muted mb-1"),
                html.H3(
                    str(value),
                    className="mb-1",
                    style={'color': self.theme.get(color, self.theme['primary'])}
                ),
                html.Small(subtitle, className="text-muted") if subtitle else None
            ])
        ], className="text-center")
    
    def create_progress_bar(self, progress: float, label: str = None) -> dbc.Progress:
        """진행률 표시 바 생성"""
        # 진행률에 따른 색상 결정
        if progress < 33:
            color = "danger"
        elif progress < 66:
            color = "warning"
        else:
            color = "success"
        
        return dbc.Progress(
            value=progress,
            color=color,
            striped=True,
            animated=True,
            label=label or f"{progress:.1f}%",
            className="mb-2"
        )
    
    def create_tab_layout(self, tabs_data: List[Dict[str, Any]]) -> dbc.Tabs:
        """탭 레이아웃 생성
        
        Parameters
        ----------
        tabs_data : List[Dict]
            각 탭의 정보 {'label': str, 'content': Any, 'tab_id': str}
        """
        tabs = []
        for tab_info in tabs_data:
            tabs.append(
                dbc.Tab(
                    label=tab_info['label'],
                    tab_id=tab_info['tab_id'],
                    children=tab_info['content']
                )
            )
        
        return dbc.Tabs(tabs, active_tab=tabs_data[0]['tab_id'] if tabs_data else None)