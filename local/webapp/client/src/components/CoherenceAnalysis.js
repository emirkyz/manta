import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

function CoherenceAnalysis({ folderData }) {
  const { coherenceScores } = folderData;

  if (!coherenceScores) {
    return (
      <Card>
        <CardContent className="p-6">
          <h3 className="text-lg font-semibold">No coherence data available</h3>
        </CardContent>
      </Card>
    );
  }

  const { topic_coherences, average_coherence } = coherenceScores;

  const coherenceData = Object.entries(topic_coherences || {}).map(([topic, score], index) => ({
    topic: topic.replace('_coherence', ''),
    score: score,
    index: index,
  }));

  const pieData = coherenceData.map((item, index) => ({
    name: item.topic,
    value: Math.abs(item.score),
    color: COLORS[index % COLORS.length],
  }));

  const getCoherenceLevel = (score) => {
    if (score > 0.1) return { level: 'Excellent', color: 'text-green-600 dark:text-green-400' };
    if (score > 0.05) return { level: 'Good', color: 'text-blue-600 dark:text-blue-400' };
    if (score > 0) return { level: 'Fair', color: 'text-yellow-600 dark:text-yellow-400' };
    if (score > -0.1) return { level: 'Poor', color: 'text-red-600 dark:text-red-400' };
    return { level: 'Very Poor', color: 'text-red-700 dark:text-red-300' };
  };

  const overallAssessment = getCoherenceLevel(average_coherence);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Coherence Analysis</h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Overall Coherence</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-3xl font-bold ${overallAssessment.color}`}>
              {average_coherence.toFixed(4)}
            </div>
            <div className={`text-lg font-medium ${overallAssessment.color}`}>
              {overallAssessment.level}
            </div>
            <p className="text-sm text-muted-foreground mt-4">
              Higher coherence scores indicate more interpretable topics.
              Positive values are generally better.
            </p>
          </CardContent>
        </Card>

        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Topic Coherence Trend</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={coherenceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="topic" />
                  <YAxis />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="score" 
                    stroke="hsl(var(--primary))" 
                    strokeWidth={2}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Coherence Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value.toFixed(3)}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Detailed Coherence Scores</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="grid grid-cols-3 gap-4 text-sm font-medium border-b pb-2">
                <div>Topic</div>
                <div className="text-right">Score</div>
                <div className="text-center">Quality</div>
              </div>
              {coherenceData
                .sort((a, b) => b.score - a.score)
                .map((row) => {
                  const assessment = getCoherenceLevel(row.score);
                  return (
                    <div key={row.topic} className="grid grid-cols-3 gap-4 text-sm py-2">
                      <div>{row.topic}</div>
                      <div className="text-right">{row.score.toFixed(4)}</div>
                      <div className="text-center">
                        <span className={assessment.color}>
                          {assessment.level}
                        </span>
                      </div>
                    </div>
                  );
                })}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Coherence Score Interpretation</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card variant="outline">
              <CardContent className="p-4">
                <h4 className="font-semibold text-green-600 dark:text-green-400">
                  Excellent (&gt; 0.1)
                </h4>
                <p className="text-sm text-muted-foreground mt-2">
                  Highly coherent topics with strong semantic relationships.
                </p>
              </CardContent>
            </Card>
            <Card variant="outline">
              <CardContent className="p-4">
                <h4 className="font-semibold text-blue-600 dark:text-blue-400">
                  Good (0.05 - 0.1)
                </h4>
                <p className="text-sm text-muted-foreground mt-2">
                  Well-defined topics with clear thematic structure.
                </p>
              </CardContent>
            </Card>
            <Card variant="outline">
              <CardContent className="p-4">
                <h4 className="font-semibold text-yellow-600 dark:text-yellow-400">
                  Fair (0 - 0.05)
                </h4>
                <p className="text-sm text-muted-foreground mt-2">
                  Moderately coherent topics that may need refinement.
                </p>
              </CardContent>
            </Card>
            <Card variant="outline">
              <CardContent className="p-4">
                <h4 className="font-semibold text-red-600 dark:text-red-400">
                  Poor (&lt; 0)
                </h4>
                <p className="text-sm text-muted-foreground mt-2">
                  Topics lack coherence and may be difficult to interpret.
                </p>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default CoherenceAnalysis;