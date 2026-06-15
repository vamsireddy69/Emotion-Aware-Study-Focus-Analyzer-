"""
Brain Router - Intelligent session analysis and suggestions system.
Analyzes session patterns and provides personalized recommendations.
"""
from collections import Counter
import numpy as np
from datetime import datetime


class BrainRouter:
    """Analyzes session data and generates intelligent suggestions."""

    def __init__(self):
        self.suggestions = []
        self.analysis = {}

    def analyze_session(self, data_buffer, duration_min, distraction_count):
        """
        Analyze entire session and generate suggestions.

        Args:
            data_buffer: List of data points with emotion, gaze, focus_score
            duration_min: Session duration in minutes
            distraction_count: Count of gaze-away events

        Returns:
            dict: Analysis results and suggestions
        """
        if not data_buffer:
            return self._empty_analysis()

        self.suggestions = []

        # Extract metrics
        focus_scores = [d['focus_score'] for d in data_buffer]
        emotions = [d['emotion'] for d in data_buffer]
        gazes = [d['gaze'] for d in data_buffer]
        noise_levels = [d.get('noise_level', 0) for d in data_buffer if 'noise_level' in d]
        switches_per_min = [d.get('switches_per_min', 0) for d in data_buffer if 'switches_per_min' in d]

        # Calculate analytics
        avg_focus = np.mean(focus_scores) if focus_scores else 0
        max_focus = np.max(focus_scores) if focus_scores else 0
        min_focus = np.min(focus_scores) if focus_scores else 0
        focus_variance = np.var(focus_scores) if focus_scores else 0
        avg_noise = np.mean(noise_levels) if noise_levels else 0
        max_noise = np.max(noise_levels) if noise_levels else 0
        avg_switch_rate = np.mean(switches_per_min) if switches_per_min else 0
        max_switch_rate = np.max(switches_per_min) if switches_per_min else 0

        emotion_counts = Counter(emotions)
        on_screen_rate = (sum(gazes) / len(gazes) * 100) if gazes else 0

        self.analysis = {
            'avg_focus': round(avg_focus, 2),
            'max_focus': max_focus,
            'min_focus': min_focus,
            'focus_variance': round(focus_variance, 2),
            'emotion_distribution': dict(emotion_counts),
            'on_screen_rate': round(on_screen_rate, 2),
            'total_samples': len(data_buffer),
            'distraction_count': distraction_count,
            'duration_min': duration_min,
            'avg_noise_level': round(avg_noise, 2),
            'max_noise_level': round(max_noise, 2),
            'avg_switch_rate': round(avg_switch_rate, 2),
            'max_switch_rate': round(max_switch_rate, 2),
        }

        # Generate suggestions
        self._suggest_focus_improvement(avg_focus, focus_variance)
        self._suggest_distraction_management(distraction_count, duration_min)
        self._suggest_emotional_well_being(emotion_counts)
        self._suggest_attention_patterns(focus_scores)
        self._suggest_productivity_tips(on_screen_rate, distraction_count)
        self._suggest_audio_environment(avg_noise, max_noise)
        self._suggest_task_focus(avg_switch_rate)

        summary = self._generate_summary(avg_focus, distraction_count, duration_min)
        self.analysis['summary'] = summary

        return {
            'analysis': self.analysis,
            'suggestions': self.suggestions,
            'summary': summary
        }

    def _empty_analysis(self):
        """Return empty analysis template."""
        return {
            'analysis': {
                'avg_focus': 0,
                'max_focus': 0,
                'min_focus': 0,
                'focus_variance': 0,
                'emotion_distribution': {},
                'on_screen_rate': 0,
                'total_samples': 0,
                'distraction_count': 0,
                'duration_min': 0,
            },
            'suggestions': ['No data to analyze. Start a new session.'],
            'summary': 'Session contains no data.'
        }

    def _suggest_focus_improvement(self, avg_focus, variance):
        """Generate focus improvement suggestions."""
        if avg_focus >= 75:
            self.suggestions.append({
                'priority': 'positive',
                'title': 'Excellent Focus!',
                'message': f'Great job maintaining {avg_focus:.0f}% focus! Keep maintaining this excellent concentration level.'
            })
        elif avg_focus >= 60:
            self.suggestions.append({
                'priority': 'info',
                'title': 'Good Focus Session',
                'message': f'Your average focus was {avg_focus:.0f}%. Consider eliminating distractions to reach 75%+.'
            })
        elif avg_focus >= 45:
            self.suggestions.append({
                'priority': 'warning',
                'title': 'Focus Below Average',
                'message': f'Focus score of {avg_focus:.0f}% suggests room for improvement. Try using the Pomodoro technique.'
            })
        else:
            self.suggestions.append({
                'priority': 'critical',
                'title': 'Low Focus Detected',
                'message': f'Average focus was only {avg_focus:.0f}%. Consider taking short breaks and reducing distractions.'
            })

        if variance > 800:
            self.suggestions.append({
                'priority': 'warning',
                'title': 'Inconsistent Focus',
                'message': 'Your focus fluctuated significantly. Try to maintain a stable work environment or take planned breaks.'
            })

    def _suggest_distraction_management(self, distraction_count, duration_min):
        """Generate distraction management suggestions."""
        if duration_min > 0:
            distractions_per_hour = (distraction_count / duration_min) * 60
        else:
            distractions_per_hour = 0

        if distraction_count == 0:
            self.suggestions.append({
                'priority': 'positive',
                'title': 'Perfect Attention!',
                'message': 'Amazing! You had zero distractions during this session.'
            })
        elif distraction_count <= 3:
            self.suggestions.append({
                'priority': 'positive',
                'title': 'Minimal Distractions',
                'message': f'Only {distraction_count} distraction(s) in {duration_min:.1f} minutes. Excellent focus control!'
            })
        elif distractions_per_hour <= 10:
            self.suggestions.append({
                'priority': 'info',
                'title': 'Manage Notifications',
                'message': f'{distraction_count} distractions detected. Silence your phone and disable notifications.'
            })
        else:
            self.suggestions.append({
                'priority': 'critical',
                'title': 'High Distraction Rate',
                'message': f'{distraction_count} distractions in {duration_min:.1f} minutes. Try a distraction-free workspace.'
            })

    def _suggest_emotional_well_being(self, emotion_counts):
        """Generate emotional well-being suggestions."""
        total = sum(emotion_counts.values())
        if total == 0:
            return

        # Find dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
        dominant_rate = (emotion_counts.get(dominant_emotion, 0) / total * 100) if dominant_emotion else 0

        emotion_suggestions = {
            'sad': {
                'priority': 'warning',
                'title': 'Mood Support',
                'message': 'Sadness detected during session. Take a break, stretch, or get some sunlight.'
            },
            'angry': {
                'priority': 'warning',
                'title': 'Stress Relief Needed',
                'message': 'Frustration was detected. Try breathing exercises or a short walk to reset.'
            },
            'fear': {
                'priority': 'warning',
                'title': 'Anxiety Support',
                'message': 'Some anxiety detected. Break tasks into smaller steps to feel more in control.'
            },
            'happy': {
                'priority': 'positive',
                'title': 'Great Mood!',
                'message': 'You maintained a positive mood throughout. Keep up this positive energy!'
            },
            'neutral': {
                'priority': 'info',
                'title': 'Neutral State',
                'message': 'You maintained a calm, neutral state. Perfect for focused work.'
            },
        }

        if dominant_emotion in emotion_suggestions:
            suggestion = emotion_suggestions[dominant_emotion].copy()
            suggestion['title'] += f' ({dominant_rate:.0f}%)'
            self.suggestions.append(suggestion)

    def _suggest_attention_patterns(self, focus_scores):
        """Generate attention pattern insights."""
        if len(focus_scores) < 4:
            return

        # Check for declining focus (fatigue)
        first_quarter = np.mean(focus_scores[:len(focus_scores)//4])
        last_quarter = np.mean(focus_scores[-len(focus_scores)//4:])

        decline = first_quarter - last_quarter

        if decline > 15:
            self.suggestions.append({
                'priority': 'warning',
                'title': 'Mental Fatigue Detected',
                'message': f'Focus declined by {decline:.0f}% during session. Take a break or engage in lighter tasks.'
            })
        elif last_quarter > first_quarter:
            self.suggestions.append({
                'priority': 'positive',
                'title': 'Improving Focus',
                'message': "Your focus improved throughout the session. You're getting into a flow state!"
            })

        # Check for peaks (flow state)
        peak_focus = max(focus_scores) if focus_scores else 0
        if peak_focus >= 85:
            self.suggestions.append({
                'priority': 'positive',
                'title': 'Flow State Achieved',
                'message': f'Peak focus of {peak_focus}% detected! This is your optimal working state.'
            })

    def _suggest_productivity_tips(self, on_screen_rate, distraction_count):
        """Generate general productivity tips."""
        if on_screen_rate >= 95:
            self.suggestions.append({
                'priority': 'positive',
                'title': 'Laser Focus',
                'message': f'{on_screen_rate:.0f}% attention on screen. Your focus technique is working well!'
            })

        if distraction_count > 0 and on_screen_rate < 80:
            self.suggestions.append({
                'priority': 'info',
                'title': 'Environment Setup',
                'message': 'Consider adjusting your workspace: better lighting, ergonomic setup, or reduced clutter.'
            })

    def _generate_summary(self, avg_focus, distraction_count, duration_min):
        """Generate a text summary of the session."""
        if avg_focus >= 75:
            focus_status = 'excellent'
        elif avg_focus >= 60:
            focus_status = 'good'
        elif avg_focus >= 45:
            focus_status = 'moderate'
        else:
            focus_status = 'poor'

        distraction_status = 'minimal' if distraction_count <= 2 else 'notable' if distraction_count <= 5 else 'significant'

        summary = (
            f'Session Summary: {duration_min:.1f} minutes with {focus_status} focus '
            f'({avg_focus:.0f}%) and {distraction_status} distractions '
            f'({distraction_count} events). '
        )

        if avg_focus >= 75 and distraction_count <= 2:
            summary += 'Excellent work maintaining focus and minimizing distractions!'
        elif avg_focus >= 60:
            summary += 'Good session overall. Small tweaks to your environment could improve results.'
        else:
            summary += 'Focus could be improved. Review suggestions and try implementing changes in your next session.'

        return summary

    def _suggest_audio_environment(self, avg_noise, max_noise):
        """Generate suggestions based on audio environment."""
        if avg_noise < 30:
            self.suggestions.append({
                'priority': 'positive',
                'title': 'Excellent Acoustic Environment',
                'message': f'Your workspace is very quiet ({avg_noise:.1f}/100). This is ideal for deep focus and concentration.'
            })
        elif avg_noise < 60:
            self.suggestions.append({
                'priority': 'info',
                'title': 'Moderate Noise Level',
                'message': f'Your environment has moderate background noise ({avg_noise:.1f}/100). Consider using noise-canceling headphones or earplugs for better focus.'
            })
        else:
            self.suggestions.append({
                'priority': 'warning',
                'title': 'High Background Noise Detected',
                'message': f'Your workspace is quite noisy ({avg_noise:.1f}/100). This can reduce focus by up to 30%. Try moving to a quieter location or using noise-blocking equipment.'
            })

    def _suggest_task_focus(self, avg_switch_rate):
        """Generate suggestions based on application/tab switching behavior."""
        if avg_switch_rate < 1:
            self.suggestions.append({
                'priority': 'positive',
                'title': 'Single-Task Master',
                'message': f'Excellent focus on one task! Avg {avg_switch_rate:.1f} switches/min. Maintain this single-tasking approach for better productivity.'
            })
        elif avg_switch_rate < 3:
            self.suggestions.append({
                'priority': 'info',
                'title': 'Moderate Task Switching',
                'message': f'You switched tasks {avg_switch_rate:.1f} times/min. This is reasonable, but try to batch your application switches to maintain flow.'
            })
        elif avg_switch_rate < 6:
            self.suggestions.append({
                'priority': 'warning',
                'title': 'High Multitasking Detected',
                'message': f'You switched apps {avg_switch_rate:.1f} times/min. This heavy context switching reduces focus by 15-30%. Try grouping related tasks.'
            })
        else:
            self.suggestions.append({
                'priority': 'critical',
                'title': 'Excessive Context Switching',
                'message': f'Very high app switching rate: {avg_switch_rate:.1f} times/min! This severely impacts focus. Close unnecessary applications and use fullscreen mode.'
            })

    def print_suggestions(self):
        """Print formatted suggestions."""
        print("\n" + '=' * 60)
        print('BRAIN ROUTING - SESSION ANALYSIS & SUGGESTIONS')
        print('=' * 60 + "\n")

        print('SESSION STATISTICS:')
        print('-' * 60)
        for key, value in self.analysis.items():
            if key == 'emotion_distribution':
                print(f'  Emotion Distribution: {value}')
            else:
                clean_key = key.replace('_', ' ').title()
                print(f'  {clean_key}: {value}')

        print("\n" + '=' * 60)
        print('PERSONALIZED SUGGESTIONS:')
        print('=' * 60 + "\n")

        if not self.suggestions:
            print('  No suggestions available.')
        else:
            for i, suggestion in enumerate(self.suggestions, 1):
                priority_color = {
                    'positive': '[+] ',
                    'info': '[i] ',
                    'warning': '[!] ',
                    'critical': '[x] '
                }.get(suggestion['priority'], '- ')

                print(f"{i}. {priority_color}{suggestion['title']}")
                print(f"   {suggestion['message']}\n")

        print('=' * 60)
        print('SESSION SUMMARY:')
        print('-' * 60)
        print(f"  {self.analysis.get('summary', 'No summary available')}")
        print('=' * 60 + "\n")
