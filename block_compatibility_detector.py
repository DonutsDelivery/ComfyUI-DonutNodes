"""
Block Compatibility Detection System for WIDEN Merge
Detects incompatible blocks that could cause blurry/low quality outputs
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import warnings
from scipy import stats
import gc

# Optional sklearn imports for advanced threshold detection
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BlockCompatibilityDetector:
    """Detects compatibility between model blocks for WIDEN merging"""
    
    def __init__(self, 
                 statistical_threshold: float = 2.0,
                 cosine_similarity_threshold: float = 0.3,
                 magnitude_ratio_threshold: float = 10.0,
                 outlier_zscore_threshold: float = 3.0):
        """
        Initialize compatibility detector
        
        Args:
            statistical_threshold: Z-score threshold for statistical difference detection
            cosine_similarity_threshold: Minimum cosine similarity between blocks
            magnitude_ratio_threshold: Maximum allowed magnitude ratio between blocks
            outlier_zscore_threshold: Z-score threshold for outlier detection
        """
        self.statistical_threshold = statistical_threshold
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.magnitude_ratio_threshold = magnitude_ratio_threshold
        self.outlier_zscore_threshold = outlier_zscore_threshold
        
        # Cache for expensive computations
        self._compatibility_cache = {}
        self._parameter_stats_cache = {}
        
    def analyze_parameter_compatibility(self, 
                                      base_param: torch.Tensor,
                                      candidate_params: List[torch.Tensor],
                                      param_name: str) -> Dict:
        """
        Analyze compatibility of parameters across models
        
        Returns:
            Dict with compatibility analysis results
        """
        results = {
            'param_name': param_name,
            'compatible_indices': [],
            'incompatible_indices': [],
            'compatibility_scores': [],
            'incompatibility_reasons': [],
            'should_merge': True,
            'filtered_params': [],
            'outlier_params': []
        }
        
        try:
            base_stats = self._get_parameter_statistics(base_param, f"{param_name}_base")
            
            if 'error' in base_stats:
                print(f"[COMPAT DEBUG] Base stats failed for {param_name}: {base_stats['error']}")
                results['should_merge'] = False
                results['incompatibility_reasons'].append(f"Base parameter statistics failed: {base_stats['error']}")
                return results
            
            for i, candidate in enumerate(candidate_params):
                if candidate.shape != base_param.shape:
                    results['incompatible_indices'].append(i)
                    results['incompatibility_reasons'].append(f"Shape mismatch: {candidate.shape} vs {base_param.shape}")
                    results['compatibility_scores'].append(0.0)
                    continue
                
                # Compute compatibility score
                compatibility_score, reasons = self._compute_compatibility_score(
                    base_param, candidate, base_stats, param_name
                )
                
                results['compatibility_scores'].append(compatibility_score)
                
                # Debug output for problematic parameters
                if compatibility_score == 0.0 and len(results['compatibility_scores']) <= 3:
                    print(f"[COMPAT DEBUG] Zero score for {param_name}, candidate {i}: {reasons}")
                
                if compatibility_score > 0.2:  # Compatible (reduced from 0.5)
                    results['compatible_indices'].append(i)
                    results['filtered_params'].append(candidate)
                else:  # Incompatible
                    results['incompatible_indices'].append(i)
                    results['incompatibility_reasons'].extend(reasons)
                    
            # Detect outliers among compatible parameters
            if len(results['filtered_params']) > 2:
                outlier_indices = self._detect_outlier_parameters(
                    base_param, results['filtered_params'], param_name
                )
                results['outlier_params'] = [results['filtered_params'][i] for i in outlier_indices]
                # Remove outliers from filtered params
                results['filtered_params'] = [
                    p for i, p in enumerate(results['filtered_params']) 
                    if i not in outlier_indices
                ]
                
            # Decision: should we merge this parameter at all?
            compatible_count = len(results['filtered_params'])
            total_count = len(candidate_params)
            
            if compatible_count == 0:
                results['should_merge'] = False
                results['incompatibility_reasons'].append("No compatible parameters found")
            elif compatible_count / total_count < 0.1:  # Less than 10% compatible (was 30%)
                results['should_merge'] = False
                results['incompatibility_reasons'].append(f"Too few compatible parameters: {compatible_count}/{total_count}")
                
        except Exception as e:
            results['should_merge'] = False
            results['incompatibility_reasons'].append(f"Analysis failed: {str(e)}")
            
        return results
    
    def _get_parameter_statistics(self, param: torch.Tensor, cache_key: str) -> Dict:
        """Get cached statistics for a parameter"""
        if cache_key in self._parameter_stats_cache:
            return self._parameter_stats_cache[cache_key]
            
        stats = {}
        try:
            param_flat = param.flatten().float()
            # Handle single-element tensors to avoid std() warning
            if param_flat.numel() <= 1:
                std_val = 0.0
            else:
                std_val = param_flat.std().item()
            
            stats = {
                'mean': param_flat.mean().item(),
                'std': std_val,
                'magnitude': torch.norm(param).item(),
                'min': param_flat.min().item(),
                'max': param_flat.max().item(),
                'abs_mean': param_flat.abs().mean().item(),
                'sparsity': (param_flat.abs() < 1e-8).float().mean().item(),
                'shape': param.shape,
                'numel': param.numel()
            }
            
            # Compute percentiles for outlier detection
            param_np = param_flat.detach().cpu().numpy()
            stats['percentiles'] = {
                'p1': np.percentile(param_np, 1),
                'p5': np.percentile(param_np, 5),
                'p25': np.percentile(param_np, 25),
                'p75': np.percentile(param_np, 75),
                'p95': np.percentile(param_np, 95),
                'p99': np.percentile(param_np, 99)
            }
            
            # Cache the results
            self._parameter_stats_cache[cache_key] = stats
            
        except Exception as e:
            print(f"[COMPAT DEBUG] Failed to compute statistics for {cache_key}: {e}")
            import traceback
            traceback.print_exc()
            stats = {'error': str(e)}
            
        return stats
    
    def _compute_compatibility_score(self, 
                                   base_param: torch.Tensor,
                                   candidate_param: torch.Tensor,
                                   base_stats: Dict,
                                   param_name: str) -> Tuple[float, List[str]]:
        """
        Compute compatibility score between base and candidate parameters
        
        Returns:
            Tuple of (compatibility_score, incompatibility_reasons)
        """
        if 'error' in base_stats:
            return 0.0, ["Base parameter statistics failed"]
            
        reasons = []
        scores = []
        
        try:
            candidate_stats = self._get_parameter_statistics(candidate_param, f"{param_name}_candidate")
            
            if 'error' in candidate_stats:
                print(f"[COMPAT DEBUG] Candidate stats failed for {param_name}: {candidate_stats['error']}")
                return 0.0, ["Candidate parameter statistics failed"]
            
            # 1. Magnitude ratio check
            base_mag = base_stats['magnitude']
            candidate_mag = candidate_stats['magnitude']
            
            if base_mag > 1e-8 and candidate_mag > 1e-8:
                mag_ratio = max(base_mag, candidate_mag) / min(base_mag, candidate_mag)
                if mag_ratio > self.magnitude_ratio_threshold:
                    reasons.append(f"Magnitude ratio too high: {mag_ratio:.2f}")
                    scores.append(0.1)
                else:
                    scores.append(1.0 - (mag_ratio - 1) / self.magnitude_ratio_threshold)
            else:
                if base_mag <= 1e-8 and candidate_mag <= 1e-8:
                    scores.append(1.0)  # Both near zero - compatible
                else:
                    reasons.append("One parameter is near zero while other is not")
                    scores.append(0.2)
            
            # 2. Cosine similarity check
            base_flat = base_param.flatten()
            candidate_flat = candidate_param.flatten()
            
            # Check for zero vectors that would cause NaN in cosine similarity
            base_norm = torch.norm(base_flat).item()
            candidate_norm = torch.norm(candidate_flat).item()
            
            if base_norm < 1e-8 or candidate_norm < 1e-8:
                # Handle zero vectors
                if base_norm < 1e-8 and candidate_norm < 1e-8:
                    cos_sim = 1.0  # Both are zero - perfectly similar
                else:
                    cos_sim = 0.0  # One is zero, one is not - completely different
                print(f"[COMPAT DEBUG] Zero vector detected for {param_name}: base_norm={base_norm:.3e}, candidate_norm={candidate_norm:.3e}")
            else:
                cos_sim = torch.cosine_similarity(base_flat, candidate_flat, dim=0).item()
                
            # Check for NaN
            if torch.isnan(torch.tensor(cos_sim)):
                print(f"[COMPAT DEBUG] NaN cosine similarity for {param_name}")
                cos_sim = 0.0
                
            if cos_sim < self.cosine_similarity_threshold:
                reasons.append(f"Low cosine similarity: {cos_sim:.3f}")
                scores.append(0.1)
            else:
                scores.append(cos_sim)
            
            # 3. Statistical distribution check
            stat_score = self._statistical_compatibility_check(base_stats, candidate_stats)
            if stat_score < 0.1:  # More lenient (was 0.3)
                reasons.append("Statistical distributions too different")
            scores.append(stat_score)
            
            # 4. Sparsity pattern check
            sparsity_diff = abs(base_stats['sparsity'] - candidate_stats['sparsity'])
            if sparsity_diff > 0.5:
                reasons.append(f"Sparsity patterns differ significantly: {sparsity_diff:.3f}")
                scores.append(0.2)
            else:
                scores.append(1.0 - sparsity_diff)
            
            # 5. Value range compatibility
            range_score = self._value_range_compatibility(base_stats, candidate_stats)
            if range_score < 0.3:
                reasons.append("Value ranges incompatible")
            scores.append(range_score)
            
            # Weighted average of all scores
            weights = [0.25, 0.25, 0.2, 0.15, 0.15]  # Prioritize magnitude and cosine similarity
            final_score = sum(s * w for s, w in zip(scores, weights))
            
            # Debug output for first few parameters
            if len(scores) != 5:
                print(f"[COMPAT DEBUG] Score count mismatch for {param_name}: expected 5, got {len(scores)}")
            
            # Check for NaN in final score
            if torch.isnan(torch.tensor(final_score)):
                print(f"[COMPAT DEBUG] NaN final score for {param_name}: scores={scores}")
                final_score = 0.0
            
            return final_score, reasons
            
        except Exception as e:
            print(f"[COMPAT DEBUG] Compatibility computation failed for {param_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0.0, [f"Compatibility computation failed: {str(e)}"]
    
    def _statistical_compatibility_check(self, base_stats: Dict, candidate_stats: Dict) -> float:
        """Check statistical compatibility between parameters"""
        try:
            # Compare means
            mean_diff = abs(base_stats['mean'] - candidate_stats['mean'])
            mean_scale = max(abs(base_stats['mean']), abs(candidate_stats['mean']), 1e-8)
            mean_score = max(0, 1 - mean_diff / mean_scale)
            
            # Compare standard deviations
            std_ratio = max(base_stats['std'], candidate_stats['std']) / max(min(base_stats['std'], candidate_stats['std']), 1e-8)
            std_score = max(0, 1 - (std_ratio - 1) / 5)  # Penalize if std differs by more than 5x
            
            # Compare absolute means
            abs_mean_diff = abs(base_stats['abs_mean'] - candidate_stats['abs_mean'])
            abs_mean_scale = max(base_stats['abs_mean'], candidate_stats['abs_mean'], 1e-8)
            abs_mean_score = max(0, 1 - abs_mean_diff / abs_mean_scale)
            
            return (mean_score + std_score + abs_mean_score) / 3
            
        except Exception as e:
            print(f"[COMPAT DEBUG] Statistical compatibility check failed: {str(e)}")
            return 0.5  # Default neutral score
    
    def _value_range_compatibility(self, base_stats: Dict, candidate_stats: Dict) -> float:
        """Check if value ranges are compatible"""
        try:
            # Compare percentile ranges
            base_range = base_stats['percentiles']['p95'] - base_stats['percentiles']['p5']
            candidate_range = candidate_stats['percentiles']['p95'] - candidate_stats['percentiles']['p5']
            
            if base_range > 1e-8 and candidate_range > 1e-8:
                range_ratio = max(base_range, candidate_range) / min(base_range, candidate_range)
                range_score = max(0, 1 - (range_ratio - 1) / 10)
            else:
                range_score = 1.0 if base_range <= 1e-8 and candidate_range <= 1e-8 else 0.2
            
            # Compare extreme values
            base_extreme = max(abs(base_stats['min']), abs(base_stats['max']))
            candidate_extreme = max(abs(candidate_stats['min']), abs(candidate_stats['max']))
            
            if base_extreme > 1e-8 and candidate_extreme > 1e-8:
                extreme_ratio = max(base_extreme, candidate_extreme) / min(base_extreme, candidate_extreme)
                extreme_score = max(0, 1 - (extreme_ratio - 1) / 20)
            else:
                extreme_score = 1.0 if base_extreme <= 1e-8 and candidate_extreme <= 1e-8 else 0.2
            
            return (range_score + extreme_score) / 2
            
        except Exception as e:
            print(f"[COMPAT DEBUG] Value range compatibility check failed: {str(e)}")
            return 0.5
    
    def _detect_outlier_parameters(self, 
                                 base_param: torch.Tensor,
                                 compatible_params: List[torch.Tensor],
                                 param_name: str) -> List[int]:
        """Detect outlier parameters among compatible ones"""
        if len(compatible_params) < 3:
            return []
        
        outlier_indices = []
        
        try:
            # Compute distance matrix for all parameters
            all_params = [base_param] + compatible_params
            distances = []
            
            for i, param1 in enumerate(all_params):
                param_distances = []
                for j, param2 in enumerate(all_params):
                    if i != j:
                        # Compute normalized distance
                        diff = (param1 - param2).flatten()
                        dist = torch.norm(diff).item()
                        norm_factor = max(torch.norm(param1).item(), torch.norm(param2).item(), 1e-8)
                        normalized_dist = dist / norm_factor
                        param_distances.append(normalized_dist)
                    else:
                        param_distances.append(0.0)
                distances.append(param_distances)
            
            # Compute average distance for each parameter (excluding base)
            avg_distances = []
            for i in range(1, len(all_params)):  # Skip base parameter
                avg_dist = np.mean(distances[i])
                avg_distances.append(avg_dist)
            
            # Find outliers using z-score
            if len(avg_distances) > 2:
                z_scores = np.abs(stats.zscore(avg_distances))
                outlier_mask = z_scores > self.outlier_zscore_threshold
                outlier_indices = np.where(outlier_mask)[0].tolist()
            
        except Exception as e:
            print(f"Warning: Outlier detection failed for {param_name}: {e}")
        
        return outlier_indices
    
    def analyze_layer_type_compatibility(self, 
                                       base_model,
                                       candidate_models: List,
                                       layer_type: str) -> Dict:
        """Analyze compatibility for all parameters of a specific layer type"""
        results = {
            'layer_type': layer_type,
            'parameter_analyses': {},
            'overall_compatibility': True,
            'compatible_model_indices': set(range(len(candidate_models))),
            'incompatible_model_indices': set(),
            'recommendations': []
        }
        
        try:
            # Get parameters for this layer type
            base_params = {}
            candidate_params = [{}] * len(candidate_models)
            
            # Collect parameters from base model
            for name, param in base_model.named_parameters():
                if self._classify_layer_type(name) == layer_type:
                    base_params[name] = param.detach().cpu().float()
            
            # Collect parameters from candidate models
            for model_idx, model in enumerate(candidate_models):
                for name, param in model.named_parameters():
                    if name in base_params:
                        if candidate_params[model_idx] == {}:
                            candidate_params[model_idx] = {}
                        candidate_params[model_idx][name] = param.detach().cpu().float()
            
            # Analyze each parameter
            for param_name, base_param in base_params.items():
                # Collect candidate parameters for this specific parameter
                param_candidates = []
                for model_idx in range(len(candidate_models)):
                    if param_name in candidate_params[model_idx]:
                        param_candidates.append(candidate_params[model_idx][param_name])
                    else:
                        param_candidates.append(None)  # Missing parameter
                
                # Filter out None values
                valid_candidates = [p for p in param_candidates if p is not None]
                valid_indices = [i for i, p in enumerate(param_candidates) if p is not None]
                
                if not valid_candidates:
                    continue
                
                # Analyze compatibility
                analysis = self.analyze_parameter_compatibility(
                    base_param, valid_candidates, param_name
                )
                
                results['parameter_analyses'][param_name] = analysis
                
                # Update overall compatibility
                if not analysis['should_merge']:
                    results['overall_compatibility'] = False
                
                # Update incompatible model indices
                incompatible_local = set(analysis['incompatible_indices'])
                incompatible_global = {valid_indices[i] for i in incompatible_local}
                results['incompatible_model_indices'].update(incompatible_global)
            
            # Update compatible model indices
            results['compatible_model_indices'] -= results['incompatible_model_indices']
            
            # Generate recommendations
            if not results['overall_compatibility']:
                results['recommendations'].append(f"Consider excluding {layer_type} layers from merge")
            
            if results['incompatible_model_indices']:
                incompatible_count = len(results['incompatible_model_indices'])
                total_count = len(candidate_models)
                results['recommendations'].append(
                    f"Exclude {incompatible_count}/{total_count} incompatible models for {layer_type} layers"
                )
            
        except Exception as e:
            results['overall_compatibility'] = False
            results['recommendations'].append(f"Analysis failed for {layer_type}: {str(e)}")
        
        return results
    
    def _classify_layer_type(self, param_name: str) -> str:
        """Classify parameter by layer type (matches DonutWidenMerge classification)"""
        name_lower = param_name.lower()
        
        if 'time_embed' in name_lower:
            return 'time_embedding'
        elif 'label_emb' in name_lower:
            return 'class_embedding'
        elif any(x in name_lower for x in ['attn', 'attention']):
            if 'cross' in name_lower:
                return 'cross_attention'
            else:
                return 'self_attention'
        elif any(x in name_lower for x in ['conv', 'convolution']):
            if 'in_layers' in name_lower or 'input' in name_lower:
                return 'input_conv'
            elif 'out_layers' in name_lower or 'output' in name_lower:
                return 'output_conv'
            elif 'skip' in name_lower or 'residual' in name_lower:
                return 'skip_conv'
            else:
                return 'feature_conv'
        elif any(x in name_lower for x in ['norm', 'group_norm', 'layer_norm']):
            return 'normalization'
        elif 'bias' in name_lower:
            return 'bias'
        elif any(x in name_lower for x in ['down', 'upsample']):
            return 'resolution_change'
        elif any(x in name_lower for x in ['proj', 'projection']):
            return 'self_attention'
        elif any(x in name_lower for x in ['to_q', 'to_k', 'to_v', 'to_out']):
            return 'self_attention'
        elif any(x in name_lower for x in ['ff', 'feedforward', 'mlp']):
            return 'self_attention'
        elif 'weight' in name_lower and 'emb' in name_lower:
            return 'time_embedding'
        else:
            return 'other'
    
    def generate_compatibility_report(self, 
                                    base_model,
                                    candidate_models: List,
                                    output_file: Optional[str] = None) -> str:
        """Generate comprehensive compatibility report"""
        report_lines = [
            "=" * 80,
            "BLOCK COMPATIBILITY ANALYSIS REPORT",
            "=" * 80,
            f"Base model: {type(base_model).__name__}",
            f"Candidate models: {len(candidate_models)}",
            f"Analysis timestamp: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
            "",
            "CONFIGURATION:",
            f"  Statistical threshold: {self.statistical_threshold}",
            f"  Cosine similarity threshold: {self.cosine_similarity_threshold}",
            f"  Magnitude ratio threshold: {self.magnitude_ratio_threshold}",
            f"  Outlier z-score threshold: {self.outlier_zscore_threshold}",
            "",
        ]
        
        # Analyze each layer type
        layer_types = ['time_embedding', 'class_embedding', 'cross_attention', 'self_attention',
                      'input_conv', 'output_conv', 'feature_conv', 'skip_conv', 'resolution_change',
                      'normalization', 'bias', 'other']
        
        overall_compatible = True
        layer_results = {}
        
        for layer_type in layer_types:
            try:
                result = self.analyze_layer_type_compatibility(base_model, candidate_models, layer_type)
                layer_results[layer_type] = result
                
                if not result['overall_compatibility']:
                    overall_compatible = False
                
                # Add to report
                report_lines.extend([
                    f"LAYER TYPE: {layer_type.upper()}",
                    "-" * 40,
                    f"  Overall compatible: {'YES' if result['overall_compatibility'] else 'NO'}",
                    f"  Compatible models: {len(result['compatible_model_indices'])}/{len(candidate_models)}",
                    f"  Parameters analyzed: {len(result['parameter_analyses'])}",
                ])
                
                if result['recommendations']:
                    report_lines.append("  Recommendations:")
                    for rec in result['recommendations']:
                        report_lines.append(f"    - {rec}")
                
                report_lines.append("")
                
            except Exception as e:
                report_lines.extend([
                    f"LAYER TYPE: {layer_type.upper()}",
                    f"  ERROR: {str(e)}",
                    ""
                ])
                overall_compatible = False
        
        # Overall summary
        report_lines.extend([
            "=" * 80,
            "OVERALL SUMMARY",
            "=" * 80,
            f"Overall compatibility: {'COMPATIBLE' if overall_compatible else 'INCOMPATIBLE'}",
            "",
            "RECOMMENDED ACTIONS:",
        ])
        
        if overall_compatible:
            report_lines.append("  ✓ Models appear compatible for WIDEN merging")
        else:
            report_lines.append("  ⚠ Compatibility issues detected:")
            
            # Aggregate recommendations
            all_recommendations = set()
            for result in layer_results.values():
                all_recommendations.update(result['recommendations'])
            
            for rec in sorted(all_recommendations):
                report_lines.append(f"    - {rec}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                print(f"Compatibility report saved to: {output_file}")
            except Exception as e:
                print(f"Warning: Failed to save report to {output_file}: {e}")
        
        return report_text
    
    def find_optimal_threshold(self, compatibility_scores: List[float], method: str = "divergence") -> float:
        """
        Automatically find the optimal compatibility threshold based on score distribution
        
        Args:
            compatibility_scores: List of compatibility scores from parameter analysis
            method: Method to use - "divergence", "kmeans", "histogram", or "gradient"
            
        Returns:
            Optimal threshold value
        """
        if not compatibility_scores or len(compatibility_scores) < 10:
            return 0.2  # Fallback for insufficient data
        
        scores = np.array(compatibility_scores)
        scores = scores[~np.isnan(scores)]  # Remove NaN values
        
        if len(scores) < 10:
            return 0.2
        
        if method == "divergence":
            return self._find_threshold_by_gap(scores)
        elif method == "kmeans":
            return self._find_threshold_by_kmeans(scores)
        elif method == "histogram":
            return self._find_threshold_by_histogram(scores)
        elif method == "gradient":
            return self._find_threshold_by_gradient(scores)
        else:
            return self._find_threshold_by_gap(scores)
    
    def _find_threshold_by_gap(self, scores: np.ndarray) -> float:
        """Find threshold at the largest gap in score distribution"""
        try:
            # Sort scores for analysis
            sorted_scores = np.sort(scores)
            n = len(sorted_scores)
            
            if n < 10:
                return 0.2
            
            # Find the largest gap between consecutive scores
            best_threshold = 0.2
            max_gap = 0
            
            # Debug info
            print(f"[GAP DEBUG] Analyzing {n} scores: min={sorted_scores[0]:.3f}, max={sorted_scores[-1]:.3f}, std={np.std(sorted_scores):.3f}")
            
            # Look for the largest gap between consecutive scores
            for i in range(1, n):
                gap = sorted_scores[i] - sorted_scores[i-1]
                
                # Only consider gaps in the middle range (not at extremes)
                if 0.1 <= sorted_scores[i-1] <= 0.9 and 0.1 <= sorted_scores[i] <= 0.9:
                    if gap > max_gap:
                        max_gap = gap
                        # Set threshold in the middle of the gap
                        best_threshold = (sorted_scores[i-1] + sorted_scores[i]) / 2
                        print(f"[GAP DEBUG] Found gap {gap:.3f} between {sorted_scores[i-1]:.3f} and {sorted_scores[i]:.3f}, threshold: {best_threshold:.3f}")
            
            # Debug final result
            print(f"[GAP DEBUG] Best threshold: {best_threshold:.3f}, max gap: {max_gap:.3f}")
            
            # If no meaningful gap found, use a more traditional approach
            if max_gap < 0.01:  # Very small gap
                print(f"[GAP DEBUG] No significant gap found, using percentile-based approach")
                # Look for natural breakpoint using percentiles
                p25 = np.percentile(sorted_scores, 25)
                p75 = np.percentile(sorted_scores, 75)
                iqr = p75 - p25
                
                # Set threshold at lower quartile minus 1 IQR (outlier detection)
                outlier_threshold = p25 - 1.5 * iqr
                if outlier_threshold > 0.1:
                    best_threshold = outlier_threshold
                    print(f"[GAP DEBUG] Using outlier threshold: {best_threshold:.3f}")
                else:
                    # Fallback to gradient method
                    gradient_threshold = self._find_threshold_by_gradient(sorted_scores)
                    if 0.1 <= gradient_threshold <= 0.8:
                        print(f"[GAP DEBUG] Using gradient threshold: {gradient_threshold:.3f}")
                        return gradient_threshold
            
            # Ensure reasonable bounds
            return max(0.1, min(0.8, best_threshold))
            
        except Exception as e:
            print(f"Warning: Gap threshold detection failed: {e}")
            return 0.2
    
    def _find_threshold_by_kmeans(self, scores: np.ndarray) -> float:
        """Find threshold using K-means clustering to separate compatible/incompatible"""
        if not SKLEARN_AVAILABLE:
            print("Warning: sklearn not available, falling back to divergence method")
            return self._find_threshold_by_gap(scores)
        
        try:
            scores_2d = scores.reshape(-1, 1)
            
            # Try K-means with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scores_2d)
            
            # Find the boundary between clusters
            cluster_centers = kmeans.cluster_centers_.flatten()
            threshold = np.mean(cluster_centers)
            
            # Validate clustering quality
            if len(set(labels)) == 2:
                silhouette = silhouette_score(scores_2d, labels)
                if silhouette > 0.3:  # Good clustering
                    return max(0.1, min(0.8, threshold))
            
            return 0.2
            
        except Exception as e:
            print(f"Warning: K-means threshold detection failed: {e}")
            return 0.2
    
    def _find_threshold_by_histogram(self, scores: np.ndarray) -> float:
        """Find threshold using histogram analysis to find the valley between peaks"""
        try:
            # Create histogram
            hist, bin_edges = np.histogram(scores, bins=20)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Smooth the histogram
            try:
                from scipy.ndimage import gaussian_filter1d
                smoothed_hist = gaussian_filter1d(hist.astype(float), sigma=1.0)
            except ImportError:
                # Simple manual smoothing if scipy not available
                smoothed_hist = hist.astype(float)
                for i in range(1, len(smoothed_hist) - 1):
                    smoothed_hist[i] = (hist[i-1] + hist[i] + hist[i+1]) / 3
            
            # Find local minima (valleys)
            valleys = []
            for i in range(1, len(smoothed_hist) - 1):
                if smoothed_hist[i] < smoothed_hist[i-1] and smoothed_hist[i] < smoothed_hist[i+1]:
                    valleys.append(i)
            
            if valleys:
                # Choose valley with best separation
                best_valley_idx = valleys[0]
                min_count = smoothed_hist[best_valley_idx]
                
                for valley_idx in valleys:
                    if smoothed_hist[valley_idx] < min_count:
                        min_count = smoothed_hist[valley_idx]
                        best_valley_idx = valley_idx
                
                threshold = bin_centers[best_valley_idx]
                return max(0.1, min(0.8, threshold))
            
            return 0.2
            
        except Exception as e:
            print(f"Warning: Histogram threshold detection failed: {e}")
            return 0.2
    
    def _find_threshold_by_gradient(self, scores: np.ndarray) -> float:
        """Find threshold using gradient analysis to find steepest change"""
        try:
            sorted_scores = np.sort(scores)
            n = len(sorted_scores)
            
            if n < 10:
                return 0.2
            
            # Compute gradient (rate of change)
            gradients = np.diff(sorted_scores)
            
            # Find the steepest increase (largest gap)
            max_gradient_idx = np.argmax(gradients)
            
            # The threshold is at the point of steepest increase
            threshold = sorted_scores[max_gradient_idx]
            
            return max(0.1, min(0.8, threshold))
            
        except Exception as e:
            print(f"Warning: Gradient threshold detection failed: {e}")
            return 0.2

    def analyze_compatibility_distribution(self, 
                                         base_model,
                                         candidate_models: List) -> Dict:
        """
        Analyze the distribution of compatibility scores to find optimal threshold
        
        Returns:
            Dict with distribution analysis and recommended threshold
        """
        all_scores = []
        parameter_analyses = {}
        
        try:
            # Collect compatibility scores from all parameters
            parameters_processed = 0
            parameters_with_scores = 0
            
            for name, param in base_model.named_parameters():
                try:
                    parameters_processed += 1
                    
                    # Get candidate parameters
                    candidates = []
                    for model in candidate_models:
                        model_params = dict(model.named_parameters())
                        if name in model_params:
                            candidates.append(model_params[name].detach().cpu().float())
                    
                    if not candidates:
                        print(f"[COMPAT DEBUG] No candidates found for parameter: {name}")
                        continue
                    
                    # Analyze compatibility for this parameter
                    analysis = self.analyze_parameter_compatibility(
                        param.detach().cpu().float(), candidates, name
                    )
                    
                    scores = analysis.get('compatibility_scores', [])
                    if scores:
                        all_scores.extend(scores)
                        parameters_with_scores += 1
                        parameter_analyses[name] = {
                            'scores': scores,
                            'should_merge': analysis.get('should_merge', True)
                        }
                        
                        # Debug output for first few parameters
                        if parameters_with_scores <= 3:
                            print(f"[COMPAT DEBUG] Parameter {name}: {len(scores)} scores, range [{min(scores):.3f}, {max(scores):.3f}]")
                    else:
                        print(f"[COMPAT DEBUG] No scores returned for parameter: {name}")
                        print(f"[COMPAT DEBUG] Analysis result: should_merge={analysis.get('should_merge', 'unknown')}")
                        if analysis.get('incompatibility_reasons'):
                            print(f"[COMPAT DEBUG] Reasons: {analysis['incompatibility_reasons'][:3]}")
                            
                except Exception as e:
                    print(f"[COMPAT DEBUG] Error analyzing parameter {name}: {str(e)}")
                    continue
            
            print(f"[COMPAT DEBUG] Processed {parameters_processed} parameters, {parameters_with_scores} had scores")
            print(f"[COMPAT DEBUG] Total scores collected: {len(all_scores)}")
            
            if not all_scores:
                return {
                    'recommended_threshold': 0.2,
                    'method': 'fallback',
                    'score_distribution': {},
                    'analysis_quality': 'poor'
                }
            
            scores_array = np.array(all_scores)
            
            # Try different threshold detection methods
            methods = {
                'divergence': self._find_threshold_by_gap(scores_array),
                'kmeans': self._find_threshold_by_kmeans(scores_array),
                'histogram': self._find_threshold_by_histogram(scores_array),
                'gradient': self._find_threshold_by_gradient(scores_array)
            }
            
            # Choose the method with most reasonable result
            reasonable_thresholds = {k: v for k, v in methods.items() 
                                   if 0.1 <= v <= 0.8}
            
            if reasonable_thresholds:
                # Use divergence method as primary, fall back to others
                recommended = methods.get('divergence', 0.2)
                if not (0.1 <= recommended <= 0.8):
                    recommended = list(reasonable_thresholds.values())[0]
            else:
                recommended = 0.2
            
            # Generate distribution statistics
            distribution_stats = {
                'mean': float(np.mean(scores_array)),
                'median': float(np.median(scores_array)),
                'std': float(np.std(scores_array)),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'q25': float(np.percentile(scores_array, 25)),
                'q75': float(np.percentile(scores_array, 75)),
                'total_parameters': len(parameter_analyses)
            }
            
            return {
                'recommended_threshold': recommended,
                'method_results': methods,
                'best_method': 'divergence',
                'score_distribution': distribution_stats,
                'analysis_quality': 'good' if len(all_scores) > 50 else 'moderate',
                'total_scores_analyzed': len(all_scores)
            }
            
        except Exception as e:
            print(f"Warning: Distribution analysis failed: {e}")
            return {
                'recommended_threshold': 0.2,
                'method': 'error_fallback',
                'score_distribution': {},
                'analysis_quality': 'error'
            }

    def clear_cache(self):
        """Clear all cached data to free memory"""
        self._compatibility_cache.clear()
        self._parameter_stats_cache.clear()
        gc.collect()

    def test_compatibility_analysis(self):
        """Test compatibility analysis with simple tensors"""
        print("\n[COMPAT TEST] Testing compatibility analysis with simple tensors...")
        
        # Create simple test tensors
        base_param = torch.randn(10, 10)
        candidate_params = [
            base_param + torch.randn(10, 10) * 0.1,  # Very similar
            base_param + torch.randn(10, 10) * 0.5,  # Moderately similar
            torch.randn(10, 10),                      # Different
            torch.zeros(10, 10),                      # Zero tensor
        ]
        
        print(f"[COMPAT TEST] Base parameter shape: {base_param.shape}")
        print(f"[COMPAT TEST] Base parameter norm: {torch.norm(base_param).item():.3f}")
        
        analysis = self.analyze_parameter_compatibility(base_param, candidate_params, "test_param")
        
        print(f"[COMPAT TEST] Analysis results:")
        print(f"  - Compatibility scores: {analysis['compatibility_scores']}")
        print(f"  - Compatible indices: {analysis['compatible_indices']}")
        print(f"  - Incompatible indices: {analysis['incompatible_indices']}")
        print(f"  - Should merge: {analysis['should_merge']}")
        print(f"  - Incompatibility reasons: {analysis['incompatibility_reasons']}")
        
        return analysis