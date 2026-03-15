import { useEffect, useState } from 'react'
import { Download, AlertCircle, Database, Check, Clock, Cpu } from 'lucide-react'
import { backendFetch } from '../lib/backend'
import { Button } from './ui/button'

interface GgufModelStatus {
  id: string
  name: string
  description: string
  downloaded: boolean
  size_bytes: number
  expected_size_bytes: number
  vram_required_gb: number
  is_text_encoder: boolean
  is_distilled: boolean
}

export function ModelManager() {
  const [ggufModels, setGgufModels] = useState<GgufModelStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [downloadingIds, setDownloadingIds] = useState<Set<string>>(new Set())
  const [vramGb, setVramGb] = useState<number | null>(null)

  const fetchStatus = async () => {
    try {
      const res = await backendFetch('/api/models/status')
      if (res.ok) {
        const data = await res.json()
        setGgufModels(data.gguf_models || [])
      }
    } catch (e) {
      console.error('Failed to fetch gguf status', e)
    } finally {
      setLoading(false)
    }
  }

  const fetchGpuInfo = async () => {
    try {
      const res = await backendFetch('/api/health')
      if (res.ok) {
        const data = await res.json()
        if (data.gpu_info?.vram) {
          // Convert MB to GB
          setVramGb(data.gpu_info.vram / 1024)
        }
      }
    } catch (e) {
      console.error('Failed to fetch health info', e)
    }
  }

  useEffect(() => {
    fetchStatus()
    fetchGpuInfo()
    const interval = setInterval(fetchStatus, 3000)
    return () => clearInterval(interval)
  }, [])

  const handleDownload = async (modelId: string) => {
    setDownloadingIds(prev => new Set(prev).add(modelId))
    try {
      await backendFetch('/api/models/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ggufModels: [modelId] })
      })
    } catch (e) {
      console.error('Download failed', e)
    }
  }

  if (loading) {
    return <div className="p-4 text-zinc-400">Loading models...</div>
  }

  const availableVram = vramGb || 24

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="p-4 shrink-0 border-b border-zinc-800">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Database className="w-5 h-5" />
          GGUF Model Manager
        </h2>
        <p className="text-zinc-400 text-sm mt-1">
          Download community-quantized models to save VRAM. Quantized models allow running LTX-Video on GPUs with less memory.
        </p>
        <div className="mt-2 text-xs text-zinc-500 bg-zinc-800/50 p-2 rounded-md border border-zinc-700/50">
          <span className="font-semibold text-zinc-300">Detected VRAM:</span> {vramGb ? `${vramGb.toFixed(1)} GB` : 'Unknown (assuming 24GB)'}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {ggufModels.length === 0 ? (
          <div className="text-zinc-500 text-center py-8">No GGUF models available in catalog.</div>
        ) : (
          ggufModels.map(model => {
            const isDownloading = downloadingIds.has(model.id)
            const isTooLarge = model.vram_required_gb > availableVram
            const isComplete = model.downloaded

            return (
              <div key={model.id} className="bg-zinc-800/40 border border-zinc-700/60 rounded-lg p-4 flex flex-col gap-3 transition-colors hover:bg-zinc-800/60">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-white font-medium flex items-center gap-2">
                      {model.name}
                      {model.is_distilled && (
                        <span className="bg-blue-500/20 text-blue-400 text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider font-bold">
                          Distilled
                        </span>
                      )}
                      {model.is_text_encoder && (
                        <span className="bg-purple-500/20 text-purple-400 text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider font-bold">
                          Text Encoder
                        </span>
                      )}
                    </h3>
                    <p className="text-zinc-400 text-sm mt-1 max-w-[85%]">{model.description}</p>
                  </div>
                  <div className="shrink-0 flex items-center gap-2">
                    {isComplete ? (
                      <span className="flex items-center gap-1 text-sm text-green-400 bg-green-400/10 px-2 py-1 rounded">
                        <Check className="w-4 h-4" /> Ready
                      </span>
                    ) : (
                      <Button
                        variant={isTooLarge ? 'secondary' : 'default'}
                        size="sm"
                        disabled={isDownloading || isTooLarge}
                        onClick={() => handleDownload(model.id)}
                        className={`text-xs ${isTooLarge ? 'opacity-50' : ''}`}
                      >
                        {isDownloading ? (
                          <><Clock className="w-3.5 h-3.5 mr-1" /> Downloading...</>
                        ) : (
                          <><Download className="w-3.5 h-3.5 mr-1" /> Download</>
                        )}
                      </Button>
                    )}
                  </div>
                </div>

                {isTooLarge && !isComplete && (
                  <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 text-red-400 text-xs p-2 rounded">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>This model requires ~{model.vram_required_gb}GB VRAM, which exceeds your detected {availableVram.toFixed(1)}GB VRAM. Expect out-of-memory errors if forced.</span>
                  </div>
                )}

                <div className="flex items-center gap-4 text-xs text-zinc-500">
                  <div className="flex items-center gap-1" title="Model file size">
                    <Database className="w-3.5 h-3.5" />
                    {(model.expected_size_bytes / 1e9).toFixed(1)} GB
                  </div>
                  <div className="flex items-center gap-1" title="VRAM Required">
                    <Cpu className="w-3.5 h-3.5" />
                    {model.vram_required_gb.toFixed(1)} GB VRAM
                  </div>
                </div>
                
                {(!isComplete && model.size_bytes > 0) && (
                  <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-1">
                    <div 
                      className="h-full bg-violet-500 transition-all duration-300"
                      style={{ width: `${Math.min(100, Math.round((model.size_bytes / model.expected_size_bytes) * 100))}%` }}
                    />
                  </div>
                )}
              </div>
            )
          })
        )}
        <details className="mt-8 border border-zinc-800 rounded-lg">
          <summary className="p-3 text-sm text-zinc-300 cursor-pointer font-medium hover:bg-zinc-800/50 rounded-lg transition-colors">
            Research & Trade-offs
          </summary>
          <div className="p-4 pt-0 text-sm text-zinc-400 space-y-2 leading-relaxed">
            <p><strong>Q2_K / Q3_K</strong>: ~8GB to 10GB VRAM. Fast and small, but noticeable degradation in fine details and motion consistency.</p>
            <p><strong>Q4_K_M / Q4_K_S</strong>: ~12GB to 13GB VRAM. <strong>Best Compromise</strong>. Highly recommended for fitting into 16GB-24GB cards while retaining ~95% of the original model's visual fidelity.</p>
            <p><strong>Q5_K_S</strong>: ~15GB VRAM. Excellent quality, good for 24GB VRAM cards if not using heavy upscalers concurrently.</p>
            <p><strong>Q6_K</strong>: ~16GB to 18GB VRAM. Near-original quality with a manageable VRAM footprint for 24GB cards.</p>
            <p><strong>Q8_0</strong>: ~20GB to 23GB VRAM. Near-perfect precision, but pushes the limits of a 24GB card.</p>
          </div>
        </details>
      </div>
    </div>
  )
}
