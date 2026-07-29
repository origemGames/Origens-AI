// Erros que indicam limite de requisições atingido (por chave/IP no
// provedor). Nesses casos, se o handler tiver um pool de chaves, vale a
// pena girar para a próxima chave e tentar de novo em vez de desistir.
function isRateLimitError(message = '') {
    const m = message.toLowerCase();
    return m.includes('429') || m.includes('rate limit') || m.includes('too many requests') || m.includes('quota');
}

// Erros temporários do lado do servidor/proxy (gateway caiu, sobrecarregado,
// ou a chamada estourou o timeout local). Diferente do erro de limite de
// tokens/requisições, aqui a chave em si não é o problema — mas vale a pena
// tentar de novo (com outra chave, se houver pool) em vez de desistir na
// primeira falha, já que esses erros costumam ser passageiros (novo v4.4).
function isTransientServerError(message = '') {
    const m = message.toLowerCase();
    return m.includes('502') || m.includes('503') || m.includes('504') ||
        m.includes('bad gateway') || m.includes('gateway timeout') || m.includes('service unavailable') ||
        m.includes('timeout') || m.includes('failed to fetch') || m.includes('network');
}

// Erro específico do limite de tokens por minuto (TPM) do provedor (ex:
// "Request too large ... tokens per minute"). Girar de chave não resolve
// isso sozinho (o limite costuma ser por conta/organização, não por chave),
// então quem chama precisa também reduzir o tamanho do prompt.
function isTokenLimitError(message = '') {
    const m = message.toLowerCase();
    return m.includes('tokens per minute') || m.includes('request too large') || (m.includes('tpm') && m.includes('limit'));
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Executa fetch com um teto de tempo (v4.4): antes, uma chamada via proxy
// podia ficar pendurada 30s+ até o provedor devolver um 504, travando toda a
// resposta. Com o timeout, desistimos mais cedo e liberamos o retry/rotação
// de chave para tentar outro caminho.
async function fetchWithTimeout(url, options = {}, timeoutMs = (typeof REQUEST_TIMEOUT_MS !== 'undefined' ? REQUEST_TIMEOUT_MS : 20000)) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } catch (e) {
        if (e.name === 'AbortError') throw new Error(`Timeout após ${timeoutMs}ms`);
        throw e;
    } finally {
        clearTimeout(timer);
    }
}

class AIHandler {
    constructor(name, config, consoleCallback) {
        this.name = name;
        this.config = config;
        this.consoleCallback = consoleCallback;
    }

    // Chave atual a usar na requisição: vem do pool (com rotação) quando
    // configurado, ou da chave fixa única quando o provedor não tem pool.
    currentKey() {
        return this.config.keyPool ? this.config.keyPool.current() : this.config.key;
    }

    async chat(messages, options = {}) {
        const startTime = Date.now();
        const poolSize = this.config.keyPool ? this.config.keyPool.size : 1;
        // Mesmo sem pool de chaves (ou com pool pequeno), erros transitórios
        // de servidor merecem pelo menos uma nova tentativa com a mesma
        // chave — 504/502/503 costumam ser passageiros (v4.4).
        const maxAttempts = Math.max(poolSize, 2);
        let lastError;

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                let response;
                if (this.name.startsWith('GEMINI')) {
                    response = await this.callGemini(messages, options);
                } else {
                    response = await this.callOpenAICompatible(messages, options);
                }
                const duration = Date.now() - startTime;
                this.consoleCallback(`[${this.name}] Sucesso em ${duration}ms${attempt > 0 ? ` (após ${attempt + 1} tentativas)` : ''}`, 'success');
                return response;
            } catch (error) {
                lastError = error;
                const rateLimited = isRateLimitError(error.message);
                const transient = isTransientServerError(error.message);
                const canRotateKey = this.config.keyPool && this.config.keyPool.size > 1 && (rateLimited || transient);
                const canRetrySameKey = !canRotateKey && transient;

                if ((canRotateKey || canRetrySameKey) && attempt < maxAttempts - 1) {
                    if (canRotateKey) {
                        this.consoleCallback(`[${this.name}] ${rateLimited ? 'Limite de requisições atingido' : 'Erro temporário de servidor'}, girando chave... (${error.message})`, 'info');
                        this.config.keyPool.rotate();
                    } else {
                        this.consoleCallback(`[${this.name}] Erro temporário de servidor, tentando novamente... (${error.message})`, 'info');
                    }
                    await sleep(400 * (attempt + 1));
                    continue;
                }
                const duration = Date.now() - startTime;
                this.consoleCallback(`[${this.name}] Erro em ${duration}ms: ${error.message}`, 'error');
                throw error;
            }
        }
        throw lastError;
    }

    async callOpenAICompatible(messages, options = {}) {
        try {
            const body = {
                model: this.config.model,
                messages: messages
            };
            if (options.maxTokens) body.max_tokens = options.maxTokens;

            const response = await fetchWithTimeout(this.config.baseUrl, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.currentKey()}`,
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'http://localhost:3000',
                    'X-Title': 'Origens AI Multi'
                },
                body: JSON.stringify(body)
            });

            if (!response.ok) {
                let errMessage = `HTTP ${response.status}`;
                try {
                    const errData = await response.json();
                    errMessage = errData.error?.message || errData.message || errMessage;
                } catch(e) {}
                throw new Error(errMessage);
            }

            const data = await response.json();
            if (data.choices && data.choices[0]?.message?.content) {
                return data.choices[0].message.content;
            } else {
                throw new Error("Resposta vazia ou inválida da API");
            }
        } catch (e) {
            if (e.message.includes('Failed to fetch') || e.message.includes('Timeout')) {
                return await this.callWithProxy(messages, options);
            }
            throw e;
        }
    }

    async callGemini(messages, options = {}) {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${this.config.model}:generateContent?key=${this.currentKey()}`;
        
        const contents = messages.map(m => ({
            role: m.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: m.content }]
        }));

        const generationConfig = {};
        if (options.maxTokens) generationConfig.maxOutputTokens = options.maxTokens;

        try {
            const response = await fetchWithTimeout(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ contents, generationConfig })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error?.message || `HTTP ${response.status}`);
            }

            const data = await response.json();
            if (data.candidates && data.candidates[0]?.content?.parts[0]?.text) {
                return data.candidates[0].content.parts[0].text;
            } else {
                throw new Error("Resposta inválida do Gemini");
            }
        } catch (e) {
            if (e.message.includes('Failed to fetch') || e.message.includes('Timeout')) {
                return await this.callGeminiWithProxy(messages, options);
            }
            throw e;
        }
    }

    // Métodos de Proxy para contornar CORS em ambiente local. Também usam
    // fetchWithTimeout (v4.4): sem isso, uma falha no proxy podia deixar a
    // chamada pendurada bem além dos ~30s vistos no console, sem chance de
    // girar de chave ou desistir de forma controlada.
    async callWithProxy(messages, options = {}) {
        const proxyUrl = "https://corsproxy.io/?";
        const targetUrl = this.config.baseUrl;

        const body = {
            model: this.config.model,
            messages: messages
        };
        if (options.maxTokens) body.max_tokens = options.maxTokens;

        const response = await fetchWithTimeout(proxyUrl + encodeURIComponent(targetUrl), {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.currentKey()}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body)
        });

        if (!response.ok) throw new Error(`Erro via Proxy: ${response.status}`);
        const data = await response.json();
        if (!data.choices || !data.choices[0]?.message?.content) throw new Error("Resposta vazia ou inválida da API (via proxy)");
        return data.choices[0].message.content;
    }

    async callGeminiWithProxy(messages, options = {}) {
        const proxyUrl = "https://corsproxy.io/?";
        const targetUrl = `https://generativelanguage.googleapis.com/v1beta/models/${this.config.model}:generateContent?key=${this.currentKey()}`;
        
        const contents = messages.map(m => ({
            role: m.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: m.content }]
        }));

        const generationConfig = {};
        if (options.maxTokens) generationConfig.maxOutputTokens = options.maxTokens;

        const response = await fetchWithTimeout(proxyUrl + encodeURIComponent(targetUrl), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contents, generationConfig })
        });

        if (!response.ok) throw new Error(`Erro Gemini via Proxy: ${response.status}`);
        const data = await response.json();
        if (!data.candidates || !data.candidates[0]?.content?.parts[0]?.text) throw new Error("Resposta vazia ou inválida do Gemini (via proxy)");
        return data.candidates[0].content.parts[0].text;
    }
}
