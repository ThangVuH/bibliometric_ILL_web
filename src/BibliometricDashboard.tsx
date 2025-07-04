import React, { useState } from 'react';
import { Play, Database, GitBranch, FileText, Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

interface LoadingState {
  [key: string]: boolean;
}

interface ResponseData {
  success: boolean;
  data: any;
  timestamp: string;
}

interface ResponseState {
  [key: string]: ResponseData;
}

interface YearRange {
  start: number;
  end: number;
}

interface Step {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  endpoint: string;
  hasParams: boolean;
}

const BibliometricDashboard: React.FC = () => {
  const [loading, setLoading] = useState<LoadingState>({});
  const [responses, setResponses] = useState<ResponseState>({});
  const [yearRange, setYearRange] = useState<YearRange>({
    start: 1970,
    end: new Date().getFullYear()
  });

  // Backend base URL - change this to match your Quart server
//   const API_BASE = 'http://localhost:5000';
  const API_BASE = 'http://127.0.0.1:5000/';

  const callEndpoint = async (endpoint: string, params: Record<string, any> = {}) => {
    const stepKey = endpoint.replace('/', '');
    setLoading(prev => ({ ...prev, [stepKey]: true }));
    
    try {
      const url = new URL(`${API_BASE}${endpoint}`);
      Object.keys(params).forEach(key => url.searchParams.append(key, params[key].toString()));
      
      const response = await fetch(url);
      const data = await response.json();
      
      setResponses(prev => ({ 
        ...prev, 
        [stepKey]: {
          success: response.ok,
          data: data,
          timestamp: new Date().toLocaleTimeString()
        }
      }));
      
    } catch (error) {
      setResponses(prev => ({ 
        ...prev, 
        [stepKey]: {
          success: false,
          data: { error: (error as Error).message },
          timestamp: new Date().toLocaleTimeString()
        }
      }));
    } finally {
      setLoading(prev => ({ ...prev, [stepKey]: false }));
    }
  };

  const steps: Step[] = [
    {
      id: 'fetch_data',
      title: 'Fetch Data',
      description: 'Fetch and store data from Flora, OpenAlex, Scopus, and Web of Science databases',
      icon: Database,
      color: 'bg-blue-500',
      endpoint: '/fetch_data',
      hasParams: true
    },
    {
      id: 'validate_data',
      title: 'Validate Data',
      description: 'Preprocess and validate data, including tier classification',
      icon: CheckCircle,
      color: 'bg-green-500',
      endpoint: '/validate_data',
      hasParams: false
    },
    {
      id: 'citation_data',
      title: 'Citation Network',
      description: 'Build citation network with D0 works and D1 citations',
      icon: GitBranch,
      color: 'bg-purple-500',
      endpoint: '/citation_data',
      hasParams: false
    },
    {
      id: 'patent_data',
      title: 'Patent Analysis',
      description: 'Analyze patent data using Lens API',
      icon: FileText,
      color: 'bg-orange-500',
      endpoint: '/patent_data',
      hasParams: false
    }
  ];

  const getStatusIcon = (stepId: string) => {
    const response = responses[stepId];
    const isLoading = loading[stepId];
    
    if (isLoading) return <Loader2 className="w-5 h-5 animate-spin text-blue-500" />;
    if (!response) return <AlertCircle className="w-5 h-5 text-gray-400" />;
    if (response.success) return <CheckCircle className="w-5 h-5 text-green-500" />;
    return <XCircle className="w-5 h-5 text-red-500" />;
  };

  const formatResponse = (data: any): string => {
    if (!data) return '';
    return JSON.stringify(data, null, 2);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Bibliometric Data Processing Pipeline
          </h1>
          <p className="text-gray-600">
            Process and analyze bibliometric data through multiple stages
          </p>
        </div>

        {/* Year Range Controls */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Configuration</h2>
          <div className="flex gap-4 items-end">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Start Year
              </label>
              <input
                type="number"
                value={yearRange.start}
                onChange={(e) => setYearRange(prev => ({ ...prev, start: parseInt(e.target.value) }))}
                className="w-24 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min={1900}
                max={new Date().getFullYear()}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                End Year
              </label>
              <input
                type="number"
                value={yearRange.end}
                onChange={(e) => setYearRange(prev => ({ ...prev, end: parseInt(e.target.value) }))}
                className="w-24 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min={1900}
                max={new Date().getFullYear()}
              />
            </div>
          </div>
        </div>

        {/* Processing Steps */}
        <div className="grid gap-6">
          {steps.map((step, index) => {
            const StepIcon = step.icon;
            const response = responses[step.id];
            const isLoading = loading[step.id];
            
            return (
              <div key={step.id} className="bg-white rounded-lg shadow-sm overflow-hidden">
                <div className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${step.color}`}>
                        <StepIcon className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">
                          Step {index + 1}: {step.title}
                        </h3>
                        <p className="text-gray-600 text-sm">{step.description}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {getStatusIcon(step.id)}
                      <button
                        onClick={() => callEndpoint(
                          step.endpoint, 
                          step.hasParams ? { 
                            start_year: yearRange.start, 
                            end_year: yearRange.end 
                          } : {}
                        )}
                        disabled={isLoading}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                          isLoading 
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                        }`}
                      >
                        {isLoading ? (
                          <Loader2 className="w-4 h-4 animate-spin" />
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                        {isLoading ? 'Processing...' : 'Run Step'}
                      </button>
                    </div>
                  </div>

                  {/* Response Display */}
                  {response && (
                    <div className="mt-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-sm font-medium text-gray-700">
                          Response ({response.timestamp}):
                        </span>
                        <span className={`text-xs px-2 py-1 rounded-full ${
                          response.success 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {response.success ? 'Success' : 'Error'}
                        </span>
                      </div>
                      
                      <div className="bg-gray-50 rounded-lg p-4 max-h-60 overflow-auto">
                        <pre className="text-sm text-gray-800 whitespace-pre-wrap">
                          {formatResponse(response.data)}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Run All Button */}
        <div className="mt-8 text-center">
          <button
            onClick={async () => {
              for (const step of steps) {
                await callEndpoint(
                  step.endpoint, 
                  step.hasParams ? { 
                    start_year: yearRange.start, 
                    end_year: yearRange.end 
                  } : {}
                );
              }
            }}
            disabled={Object.values(loading).some(Boolean)}
            className={`px-8 py-3 rounded-lg font-medium transition-colors ${
              Object.values(loading).some(Boolean)
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:from-blue-700 hover:to-purple-700'
            }`}
          >
            {Object.values(loading).some(Boolean) ? 'Processing...' : 'Run All Steps'}
          </button>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-500 text-sm">
          <p>Make sure your Quart backend is running on {API_BASE}</p>
        </div>
      </div>
    </div>
  );
};

export default BibliometricDashboard;