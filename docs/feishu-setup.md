# 用你自己的飞书号发消息：配置步骤

jms-top 要**以你的名义**给同事发协商消息，而不是以一个机器人名义。群里的人看到的是你，
才谈得上协商 —— 机器人发的消息没人会当成一个需要回应的请求。

这条路要用飞书**自建应用**和**用户身份令牌（`user_access_token`）**，比群机器人 webhook
麻烦，但它也是唯一能做到"用我的号发"的方式。

---

## 一、建应用（约 5 分钟）

1. 打开 <https://open.feishu.cn/app>，点 **「创建企业自建应用」**。
2. 名称随意，比如 `jms-top 协商助手`。图标可以跳过。
3. 创建后进入应用详情，在 **「凭证与基础信息」** 页记下两个值：

   ```
   App ID      cli_xxxxxxxxxxxxxxxx
   App Secret  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

   > App Secret 等同于这个应用的密码，别贴到公开的地方，也别提交进 git。

## 二、开权限（可能要管理员审批）

进入 **「权限管理」**，搜索并添加下面这些。**注意每个权限都要选「用户身份」** ——
选成「应用身份」的话，消息会以机器人名义发出，就白做了。

| 权限码 | 用途 |
|---|---|
| `im:message:send_as_bot` | 发送消息（部分企业要求此项） |
| `im:message` | 读写消息 |
| `contact:user.base:readonly` | 需要按名字查你的 open_id 时用 |

加完点 **「创建版本并发布」**。如果是企业管理员管控的权限，会进入审批，等通过即可。

## 三、配置重定向 URL

进入 **「安全设置」** → **「重定向 URL」**，添加：

```
http://localhost:8765/callback
```

这是 jms-top 本地接收授权码用的。端口 `8765` 是我定的，配置里可改。

## 四、把凭据填进配置

```toml
# ~/.config/jms-top/config.toml

[feishu]
app_id = "cli_xxxxxxxxxxxxxxxx"
app_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
redirect_port = 8765          # 与上面重定向 URL 的端口一致
```

填完执行：

```bash
jms-top feishu login
```

它会：

1. 打印一个飞书授权链接，你在浏览器打开；
2. 你在飞书里点「授权」；
3. 浏览器跳回 `localhost:8765`，jms-top 拿到授权码；
4. 换取 `user_access_token` 和 `refresh_token`，存进 `~/.config/jms-top/feishu-token.json`（0600）。

## 五、第一次发消息前，设置收件人

```bash
jms-top feishu whoami            # 确认现在是你的账号
jms-top feishu recipient 张三     # 按名字查人，存下 open_id
```

之后 `n` 生成草稿、`s` 发送，飞书里显示的就是你自己发出的消息。

---

## 关于令牌有效期

| 令牌 | 有效期 | 过期后 |
|---|---|---|
| `user_access_token` | **约 2 小时** | 自动用 refresh_token 换新的 |
| `refresh_token` | **约 30 天** | 需要重新 `jms-top feishu login` |

jms-top 会在每次发送前检查并自动刷新，你不用管 2 小时这件事。但如果 **30 天**没用过，
就需要重新授权一次 —— 届时发送会失败并提示你重新登录，不会静默失效。

---

## 已验证 / 待验证

写这份文档时，我实际做了网络探测，下面这些是**真实响应**，不是照文档抄的：

| 检查项 | 结果 |
|---|---|
| `open.feishu.cn` API 可达 | ✅ 可达（文档网页被 DNS 拦截，API 没有） |
| OAuth 端点存在且返回结构化错误 | ✅ `{"error":"invalid_grant","code":20003}` |
| 发消息端点存在且要求令牌 | ✅ `{"code":99991661,"msg":"Missing access token..."}` |

**还没有验证的**（需要你的 App ID/Secret 才能测）：

- 完整的授权码换取令牌流程；
- 拿到令牌后能否真正发出一条消息；
- 你的企业是否需要额外审批。

也就是说：**这个方案在"能不能发出去"这一点上，我只验证到了"端点是对的"，没验证到"能发成功"。**
拿到你的应用凭据后我会立刻跑通全流程，并把结论补到这里。