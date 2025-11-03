import { useState, useEffect } from "react";
import "@/App.css";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Activity, TrendingUp, Server, Clock, Zap, Calendar, CalendarDays } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Toaster, toast } from 'sonner';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Calendar as CalendarComponent } from '@/components/ui/calendar';
import { format } from 'date-fns';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [predictions, setPredictions] = useState([]);
  const [selectedModel, setSelectedModel] = useState('catboost');
  const [loading, setLoading] = useState(false);
  const [scalingResult, setScalingResult] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [calendarOpen, setCalendarOpen] = useState(false);
  const [currentMetrics, setCurrentMetrics] = useState({
    avgLoad: 0,
    peakLoad: 0,
    recommendedInstances: 0,
    festivals: 0
  });

  // Fetch 24-hour predictions
  const fetchPredictions = async (modelName, startDate = null) => {
    setLoading(true);
    try {
      const predictDate = startDate || selectedDate;
      const response = await axios.post(`${API}/predict`, {
        start_time: predictDate.toISOString(),
        hours: 24,
        model_name: modelName
      });
      
      const data = response.data;
      setPredictions(data);
      
      // Calculate metrics
      const loads = data.map(p => p.predicted_load);
      const avgLoad = loads.reduce((a, b) => a + b, 0) / loads.length;
      const peakLoad = Math.max(...loads);
      const festivals = data.filter(p => p.is_festival === 1).length;
      
      let recommendedInstances = 2;
      if (peakLoad > 5000) recommendedInstances = 10;
      else if (peakLoad > 3000) recommendedInstances = 5;
      else if (peakLoad > 1500) recommendedInstances = 3;
      
      setCurrentMetrics({
        avgLoad: Math.round(avgLoad),
        peakLoad: Math.round(peakLoad),
        recommendedInstances,
        festivals
      });
      
      toast.success(`Predictions loaded for ${format(predictDate, 'MMM dd, yyyy')} with ${modelName.toUpperCase()}`);
    } catch (error) {
      console.error('Error fetching predictions:', error);
      toast.error('Failed to fetch predictions');
    } finally {
      setLoading(false);
    }
  };

  // Trigger AWS scaling
  const triggerScaling = async () => {
    if (predictions.length === 0) {
      toast.error('Generate predictions first');
      return;
    }
    
    try {
      const peakPrediction = Math.max(...predictions.map(p => p.predicted_load));
      const response = await axios.post(`${API}/scale`, {
        predicted_load: peakPrediction,
        asg_name: 'my-web-asg'
      });
      
      setScalingResult(response.data);
      
      if (response.data.mode === 'mock') {
        toast.info('Mock scaling (Add AWS credentials for real scaling)');
      } else {
        toast.success('AWS Auto Scaling triggered!');
      }
    } catch (error) {
      console.error('Error triggering scaling:', error);
      toast.error('Failed to trigger scaling');
    }
  };

  // Handle model change
  const handleModelChange = (model) => {
    setSelectedModel(model);
    fetchPredictions(model, selectedDate);
  };

  // Handle date change
  const handleDateChange = (date) => {
    if (date) {
      setSelectedDate(date);
      setCalendarOpen(false);
      fetchPredictions(selectedModel, date);
    }
  };

  // Load predictions on mount
  useEffect(() => {
    fetchPredictions('catboost', new Date());
  }, []);

  // Prepare chart data
  const chartData = predictions.map((pred, idx) => ({
    hour: `${pred.hour}:00`,
    load: Math.round(pred.predicted_load),
    isFestival: pred.is_festival === 1,
    festivalName: pred.festival_name
  }));

  return (
    <div className="app-container" data-testid="app-container">
      <Toaster position="top-right" richColors />
      
      {/* Header */}
      <header className="app-header" data-testid="app-header">
        <div className="header-content">
          <div className="header-left">
            <div className="logo-container">
              <Zap className="logo-icon" />
              <h1 className="app-title">AI Predictive Autoscaling</h1>
            </div>
            <p className="app-subtitle">Real-time traffic prediction with intelligent EC2 scaling</p>
          </div>
          
          <div className="header-right">
            <div className="model-selector">
              <label className="model-label">ML Model:</label>
              <Select value={selectedModel} onValueChange={handleModelChange}>
                <SelectTrigger className="model-trigger" data-testid="model-selector">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="catboost">CatBoost</SelectItem>
                  <SelectItem value="lightgbm">LightGBM</SelectItem>
                  <SelectItem value="xgboost">XGBoost</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Metrics Cards */}
        <div className="metrics-grid" data-testid="metrics-grid">
          <Card className="metric-card">
            <CardHeader className="metric-header">
              <Activity className="metric-icon" style={{ color: '#3b82f6' }} />
              <CardTitle className="metric-title">Avg Load</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="metric-value">{currentMetrics.avgLoad}</div>
              <div className="metric-label">requests/hour</div>
            </CardContent>
          </Card>

          <Card className="metric-card">
            <CardHeader className="metric-header">
              <TrendingUp className="metric-icon" style={{ color: '#ef4444' }} />
              <CardTitle className="metric-title">Peak Load</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="metric-value">{currentMetrics.peakLoad}</div>
              <div className="metric-label">requests/hour</div>
            </CardContent>
          </Card>

          <Card className="metric-card">
            <CardHeader className="metric-header">
              <Server className="metric-icon" style={{ color: '#10b981' }} />
              <CardTitle className="metric-title">Recommended</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="metric-value">{currentMetrics.recommendedInstances}</div>
              <div className="metric-label">EC2 instances</div>
            </CardContent>
          </Card>

          <Card className="metric-card">
            <CardHeader className="metric-header">
              <Calendar className="metric-icon" style={{ color: '#f59e0b' }} />
              <CardTitle className="metric-title">Festivals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="metric-value">{currentMetrics.festivals}</div>
              <div className="metric-label">in next 24h</div>
            </CardContent>
          </Card>
        </div>

        {/* Prediction Chart */}
        <Card className="chart-card" data-testid="prediction-chart">
          <CardHeader>
            <CardTitle className="chart-title">
              <Clock className="chart-icon" />
              24-Hour Traffic Prediction
            </CardTitle>
            <CardDescription>Hourly traffic forecast for optimal autoscaling</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="loading-state">
                <div className="spinner"></div>
                <p>Loading predictions...</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="loadGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis 
                    dataKey="hour" 
                    stroke="#6b7280"
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis 
                    stroke="#6b7280"
                    tick={{ fontSize: 12 }}
                    label={{ value: 'Traffic Load', angle: -90, position: 'insideLeft', style: { fontSize: 12, fill: '#6b7280' } }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
                    }}
                    content={({ active, payload }) => {
                      if (active && payload && payload[0]) {
                        const data = payload[0].payload;
                        return (
                          <div className="custom-tooltip">
                            <p className="tooltip-hour">{data.hour}</p>
                            <p className="tooltip-load">Load: <strong>{data.load}</strong></p>
                            {data.isFestival && (
                              <Badge className="festival-badge">{data.festivalName}</Badge>
                            )}
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="load" 
                    stroke="#3b82f6" 
                    strokeWidth={3}
                    fill="url(#loadGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        {/* Actions & Scaling */}
        <div className="actions-section">
          <Card className="action-card">
            <CardHeader>
              <CardTitle className="action-title">AWS Auto Scaling Control</CardTitle>
              <CardDescription>Trigger EC2 instance scaling based on predictions</CardDescription>
            </CardHeader>
            <CardContent>
              <Button 
                onClick={triggerScaling}
                disabled={loading || predictions.length === 0}
                className="scale-button"
                data-testid="trigger-scaling-btn"
              >
                <Server className="button-icon" />
                Trigger AWS Scaling
              </Button>
              
              {scalingResult && (
                <div className="scaling-result" data-testid="scaling-result">
                  <div className="result-header">
                    {scalingResult.mode === 'mock' ? (
                      <Badge variant="outline" className="mode-badge mode-mock">Mock Mode</Badge>
                    ) : (
                      <Badge variant="default" className="mode-badge mode-real">Real AWS</Badge>
                    )}
                  </div>
                  <div className="result-content">
                    <div className="result-item">
                      <span className="result-label">Predicted Load:</span>
                      <span className="result-value">{Math.round(scalingResult.predicted_load)}</span>
                    </div>
                    <div className="result-item">
                      <span className="result-label">Desired Capacity:</span>
                      <span className="result-value">{scalingResult.desired_capacity} instances</span>
                    </div>
                    <p className="result-message">{scalingResult.message}</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="info-card">
            <CardHeader>
              <CardTitle className="info-title">Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="info-list">
                <div className="info-item">
                  <span className="info-key">Model:</span>
                  <span className="info-value">{selectedModel.toUpperCase()}</span>
                </div>
                <div className="info-item">
                  <span className="info-key">Prediction Window:</span>
                  <span className="info-value">24 hours</span>
                </div>
                <div className="info-item">
                  <span className="info-key">Festival API:</span>
                  <span className="info-value">Calendarific</span>
                </div>
                <div className="info-item">
                  <span className="info-key">AWS Service:</span>
                  <span className="info-value">EC2 Auto Scaling</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Festival Timeline */}
        {predictions.filter(p => p.is_festival === 1).length > 0 && (
          <Card className="festival-card" data-testid="festival-timeline">
            <CardHeader>
              <CardTitle className="festival-title">
                <Calendar className="festival-icon" />
                Upcoming Festivals
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="festival-list">
                {predictions
                  .filter(p => p.is_festival === 1)
                  .map((pred, idx) => (
                    <div key={idx} className="festival-item">
                      <Badge className="festival-time-badge">{pred.hour}:00</Badge>
                      <span className="festival-name">{pred.festival_name}</span>
                      <span className="festival-load">Expected: {Math.round(pred.predicted_load)} req/h</span>
                    </div>
                  ))
                }
              </div>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}

export default App;
