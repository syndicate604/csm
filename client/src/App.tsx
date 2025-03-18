import { useState, useRef, useEffect, useCallback } from 'react';
import { Box, Button, Group, LoadingOverlay, NumberInput, Progress, Slider, Stack, Text, Title, MantineProvider, SegmentedControl, Center, createTheme } from '@mantine/core';
import { IconMan, IconWoman, IconRobot, IconDownload } from '@tabler/icons-react';
import axios from 'axios';
import '@mantine/core/styles.css';
import './styles.css';

// Create theme with TypeScript type safety
const theme = createTheme({
  primaryColor: 'blue',
  defaultRadius: 'md',
  fontFamily: 'Inter, sans-serif',
  components: {
    Button: {
      defaultProps: {
        size: 'md',
        radius: 'md',
      },
    },
    Progress: {
      defaultProps: {
        size: 'md',
        radius: 'xl',
        color: 'blue',
        striped: true,
      },
    },
    LoadingOverlay: {
      defaultProps: {
        blur: 2,
        zIndex: 1000,
      },
    },
    NumberInput: {
      defaultProps: {
        size: 'md',
        radius: 'md',
      },
    },
    Slider: {
      defaultProps: {
        size: 'md',
        radius: 'xl',
        color: 'blue',
      },
    },
  },
});

export default function App() {
  const getInitialSpeaker = () => {
    const saved = localStorage.getItem('selectedSpeaker');
    return saved !== null ? Number(saved) : 0;
  };

  const [message, setMessage] = useState('');
  const [speaker, setSpeaker] = useState<number>(getInitialSpeaker());
  const [temperature, setTemperature] = useState(0.9);
  const [maxAudioLength, setMaxAudioLength] = useState(10);  // Store in seconds
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [totalDuration, setTotalDuration] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Save preferences when changed
  useEffect(() => {
    localStorage.setItem('selectedSpeaker', speaker.toString());
  }, [speaker]);

  useEffect(() => {
    localStorage.setItem('temperature', temperature.toString());
  }, [temperature]);

  useEffect(() => {
    localStorage.setItem('maxAudioLength', maxAudioLength.toString());
  }, [maxAudioLength]);

  // Cleanup audio URL on unmount
  useEffect(() => {
    const cleanup = (): void => {
      try {
        if (audioUrl) {
          URL.revokeObjectURL(audioUrl);
        }
      } catch (error) {
        console.error('Error cleaning up audio URL:', error);
      }
    };
    return cleanup;
  }, [audioUrl]);

  // Cleanup timer on unmount
  useEffect(() => {
    const cleanup = (): void => {
      try {
        if (timerRef.current) {
          window.clearInterval(timerRef.current);
          timerRef.current = null;
        }
      } catch (error) {
        console.error('Error cleaning up timer:', error);
      }
    };
    return cleanup;
  }, []);

  // Update elapsed time during audio playback
  const updateElapsedTime = useCallback(() => {
    if (audioRef.current) {
      setElapsedTime(audioRef.current.currentTime);
    }
  }, []);

  // Start timer for audio playback
  const startTimer = useCallback(() => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
    }
    timerRef.current = window.setInterval(updateElapsedTime, 100);
  }, [updateElapsedTime]);

  // Stop timer for audio playback
  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setElapsedTime(0);
  }, []);

  const handleSubmit = async () => {
    if (!message.trim()) return;

    setIsLoading(true);
    setProgress(0);
    setProgressText('Initializing...');
    stopTimer();

    try {
      // Calculate estimated duration based on text length
      const wordsCount = message.trim().split(/\s+/).length;
      const charsCount = message.trim().length;
      const requestDuration = Math.min(maxAudioLength * 1000, Math.max(
        2000, // minimum 2 seconds
        (charsCount * 100) + // 100ms per character for natural speech
        (wordsCount * 300) + // 300ms per word for pauses
        1000 // 1 second buffer
      ));

      const response = await axios.post('http://localhost:8000/chat', {
        message: message.trim(),
        speaker,
        temperature,
        max_audio_length_ms: requestDuration,
      }, {
        responseType: 'blob',
        onDownloadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percent = Math.round((progressEvent.loaded / progressEvent.total) * 100);
            setProgress(percent);
            setProgressText(`Generating audio... ${percent}%`);
          }
        },
      });

      // Create blob URL for audio playback
      const blob = new Blob([response.data], { type: 'audio/mpeg' });
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);

      // Load and play audio
      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.load();
        setProgressText('Click play to start audio playback');
      }
    } catch (error) {
      console.error('Error:', error);
      setProgressText('Error generating audio. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle audio events
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handlePlay = () => {
      startTimer();
      setProgressText('Playing audio...');
    };

    const handlePause = () => {
      stopTimer();
      setProgressText('Audio paused');
    };

    const handleEnded = () => {
      stopTimer();
      setProgressText('Audio finished');
      setElapsedTime(0);
    };

    const handleLoadedMetadata = () => {
      setTotalDuration(audio.duration);
    };

    const handleTimeUpdate = () => {
      setElapsedTime(audio.currentTime);
    };

    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      stopTimer();
    };
  }, [startTimer, stopTimer]);

  const formatFilename = (text: string): string => {
    // Remove special characters and replace spaces with underscores
    return text
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')  // Remove special chars except spaces and hyphens
      .replace(/\s+/g, '_')          // Replace spaces with underscores
      .replace(/-+/g, '-')           // Replace multiple hyphens with single hyphen
      .slice(0, 50);                 // Limit length to 50 chars
  };

  const handleDownload = () => {
    if (audioUrl && message) {
      const link = document.createElement('a');
      link.href = audioUrl;
      link.download = `${formatFilename(message)}.mp3`;
      link.click();
    }
  };

  // Format time for display
  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const decimal = Math.floor((ms % 1000) / 10).toString().padStart(2, '0');  // Get hundredths
    return `${seconds}.${decimal}s`;
  };

  return (
    <MantineProvider theme={theme}>
      <Box 
        p="xl" 
        maw={800} 
        mx="auto" 
        display="flex" 
        style={{ 
          minHeight: '100vh',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative'
        }}
      >
        <LoadingOverlay 
          visible={isLoading} 
          zIndex={1000} 
          overlayProps={{ blur: 2 }}
          loaderProps={{ size: 'lg' }}
        />

        <Stack gap="md" w="100%">
          <Title ta="center">CSM Chat</Title>

          <Box w="100%">
            <Text size="sm" mb={4}>Message</Text>
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Enter your message here..."
              className="message-input"
              rows={4}
              disabled={isLoading}
            />
          </Box>

          {isLoading && (
            <Box w="100%">
              <Text size="sm" mb={4}>Progress</Text>
              <Progress 
                value={progress} 
                size="md"
                radius="xl" 
                color="blue" 
                striped 
                animated={progress < 100}
              />
            </Box>
          )}

          <Box w="100%">
            <Text size="sm" mb={4}>Speaker Voice</Text>
            <SegmentedControl
              w="100%"
              data={[
                {
                  value: "0",
                  label: (
                    <Center>
                      <IconMan size="1.2rem" />
                      <Box ml={10}>Male</Box>
                      <Text size="xs" c="dimmed" ml={4}>Natural</Text>
                    </Center>
                  ),
                },
                {
                  value: "1",
                  label: (
                    <Center>
                      <IconRobot size="1.2rem" />
                      <Box ml={10}>Made</Box>
                      <Text size="xs" c="dimmed" ml={4}>Synthetic</Text>
                    </Center>
                  ),
                },
                {
                  value: "2",
                  label: (
                    <Center>
                      <IconWoman size="1.2rem" />
                      <Box ml={10}>Female</Box>
                      <Text size="xs" c="dimmed" ml={4}>Natural</Text>
                    </Center>
                  ),
                },
              ]}
              value={speaker.toString()}
              onChange={(value) => setSpeaker(Number(value))}
            />
          </Box>

          <Box w="100%">
            <Text size="sm" mb={4}>Temperature</Text>
            <Text size="xs" c="dimmed" mb={4}>Higher values make the output more creative</Text>
            <NumberInput
              min={0.1}
              max={1.0}
              step={0.1}
              value={temperature}
              onChange={(value) => setTemperature(Number(value) || 0.9)}
              w="100%"
              size="md"
              radius="md"
              clampBehavior="strict"
              allowDecimal
              hideControls
            />
          </Box>

          <Box w="100%">
            <Text size="sm" mb={4}>Max Audio Length (seconds)</Text>
            <Text size="xs" c="dimmed" mb={4}>Longer text needs more time to generate fully. Audio will be encoded at 128kbps mono for optimal quality.</Text>
            <Slider
              min={3}
              max={30}
              step={1}
              value={maxAudioLength}
              onChange={setMaxAudioLength}
              marks={[
                { value: 3, label: '3s' },
                { value: 10, label: '10s' },
                { value: 20, label: '20s' },
                { value: 30, label: '30s' },
              ]}
              size="md"
              radius="xl"
              color="blue"
              label={(value) => `${value}s`}
            />
          </Box>

          <Box w="100%" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px 0', gap: '10px' }}>
            <Button 
              onClick={handleSubmit} 
              loading={isLoading}
              disabled={!message.trim()}
              size="md"
              radius="md"
              px={40}
              fullWidth
            >
              Generate Audio
            </Button>
            {isLoading && (
              <Text size="sm" c="dimmed">{progressText}</Text>
            )}
          </Box>

          {audioUrl && (
            <Box w="100%">
              <Group w="100%" justify="space-between" mb={8}>
                <Text size="sm">Audio Playback</Text>
                <Text size="sm" c="dimmed">{formatTime(elapsedTime * 1000)} / {formatTime(totalDuration * 1000)}</Text>
              </Group>
              <Box 
                style={{ 
                  border: '1px solid var(--mantine-color-gray-3)', 
                  borderRadius: 'var(--mantine-radius-md)',
                  padding: '10px',
                  marginBottom: '1rem',
                  backgroundColor: 'var(--mantine-color-gray-0)'
                }}
              >
                <audio
                  ref={audioRef}
                  controls
                  preload="auto"
                  style={{ width: '100%' }}
                />
              </Box>
              <Group w="100%" gap="md">
                <Button 
                  variant="light" 
                  onClick={handleDownload} 
                  size="md" 
                  radius="md"
                  leftSection={<IconDownload size="1rem" />}
                  fullWidth
                >
                  Download MP3 (128kbps Mono)
                </Button>
              </Group>
            </Box>
          )}
        </Stack>
      </Box>
    </MantineProvider>
  );
}
