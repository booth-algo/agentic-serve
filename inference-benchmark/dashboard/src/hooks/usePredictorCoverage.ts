import { useEffect, useState } from 'react';
import { predictorCoverageJsonUrl } from '../dataUrls';
import type { PredictorCoverage } from '../types-predictor-coverage';

export function usePredictorCoverage() {
  const [predictorCoverage, setPredictorCoverage] = useState<PredictorCoverage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(predictorCoverageJsonUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: PredictorCoverage) => {
        setPredictorCoverage(json);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return { predictorCoverage, loading, error };
}
